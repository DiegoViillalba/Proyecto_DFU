#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

# Define text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}           PROYECTO DFU - CONFIGURACIÓN DE ENTORNO           ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Get directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Function to check python version
check_python_version() {
    local python_cmd=$1
    if ! command -v "$python_cmd" &> /dev/null; then
        return 1
    fi
    local version=$("$python_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$version" == "3.11" ]; then
        return 0
    else
        return 2
    fi
}

# Parse arguments or auto-detect
METHOD=""
if [ "$1" == "--conda" ]; then
    METHOD="conda"
elif [ "$1" == "--venv" ] || [ "$1" == "--pip" ]; then
    METHOD="venv"
else
    # Check if run in an interactive terminal
    if [ -t 0 ]; then
        echo -e "${YELLOW}Seleccione el método de instalación:${NC}"
        echo "1) Conda (Recomendado - Crea entorno 'dfu_env' con PyTorch GPU)"
        echo "2) Python venv + pip (Instala via pip con CUDA 12.1 soporte)"
        read -rp "Ingrese su opción [1-2]: " choice
        if [ "$choice" == "1" ]; then
            METHOD="conda"
        elif [ "$choice" == "2" ]; then
            METHOD="venv"
        else
            echo -e "${RED}[-] Opción no válida. Saliendo...${NC}"
            exit 1
        fi
    else
        # Non-interactive fallback
        if command -v conda &> /dev/null; then
            METHOD="conda"
            echo -e "${BLUE}[+] Modo no interactivo: Seleccionado Conda automáticamente porque está instalado.${NC}"
        else
            METHOD="venv"
            echo -e "${BLUE}[+] Modo no interactivo: Seleccionado Python venv automáticamente porque Conda no está instalado.${NC}"
        fi
    fi
fi

if [ "$METHOD" == "conda" ]; then
    echo -e "\n${BLUE}[+] Buscando Conda...${NC}"
    if command -v conda &> /dev/null; then
        echo -e "${GREEN}[+] Conda detectado.${NC}"
        echo -e "${BLUE}[+] Creando el entorno conda 'dfu_env' a partir de environment.yml...${NC}"
        conda env create -f environment.yml --force
        
        echo -e "${GREEN}[+] Entorno conda 'dfu_env' creado con éxito.${NC}"
        echo -e "${YELLOW}[!] Para activar este entorno, ejecute: conda activate dfu_env${NC}"
        
        # Run check script in the new conda environment
        echo -e "\n${BLUE}[+] Ejecutando verificación de instalación en el entorno conda...${NC}"
        conda run -n dfu_env python scripts/check_installation.py
    else
        echo -e "${RED}[-] Error: Conda no está instalado o no se encuentra en el PATH.${NC}"
        echo "Por favor instale Miniconda/Anaconda o elija el modo venv (--venv)."
        exit 1
    fi

elif [ "$METHOD" == "venv" ]; then
    echo -e "\n${BLUE}[+] Configurando entorno Python virtual (venv)...${NC}"
    
    # Try to find python3.11 or python
    PYTHON_CMD=""
    if check_python_version "python3.11"; then
        PYTHON_CMD="python3.11"
    elif check_python_version "python3"; then
        PYTHON_CMD="python3"
    elif check_python_version "python"; then
        PYTHON_CMD="python"
    else
        # If python version is not 3.11, search for any python and warn
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
            echo -e "${YELLOW}[!] Advertencia: Se usará la versión actual de Python ($($PYTHON_CMD -V)). Se recomienda Python 3.11.7.${NC}"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
            echo -e "${YELLOW}[!] Advertencia: Se usará la versión actual de Python ($($PYTHON_CMD -V)). Se recomienda Python 3.11.7.${NC}"
        else
            echo -e "${RED}[-] Error: No se encontró una instalación de Python.${NC}"
            exit 1
        fi
    fi

    echo -e "${BLUE}[+] Usando comando de Python: $PYTHON_CMD (${NC}$($PYTHON_CMD -V)${BLUE})${NC}"
    
    # Create venv
    echo -e "${BLUE}[+] Creando entorno virtual en '.venv'...${NC}"
    $PYTHON_CMD -m venv .venv
    
    # Activate virtual environment
    echo -e "${BLUE}[+] Activando entorno virtual...${NC}"
    source .venv/bin/activate
    
    # Upgrade pip
    echo -e "${BLUE}[+] Actualizando pip y setuptools...${NC}"
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    echo -e "${BLUE}[+] Instalando dependencias desde requirements.txt...${NC}"
    pip install -r requirements.txt
    
    echo -e "${GREEN}[+] Dependencias instaladas correctamente.${NC}"
    
    # Run verification script
    echo -e "\n${BLUE}[+] Ejecutando verificación de instalación...${NC}"
    python scripts/check_installation.py
    
    echo -e "${YELLOW}[!] Para activar este entorno virtual en su terminal, ejecute: source .venv/bin/activate${NC}"
fi
