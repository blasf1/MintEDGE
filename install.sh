#!/bin/bash
# MintEDGE Interactive Installation Script for Linux
# Compatible with Ubuntu/Debian-based distributions

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

ask_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Banner
clear
echo -e "${BLUE}"
cat << "EOF"
  __  __ _       _   _____ ____   ____ _____ 
 |  \/  (_)_ __ | |_| ____|  _ \ / ___| ____|
 | |\/| | | '_ \| __|  _| | | | | |  _|  _|  
 | |  | | | | | | |_| |___| |_| | |_| | |___ 
 |_|  |_|_|_| |_|\__|_____|____/ \____|_____|
                                              
    Interactive Installation Script
EOF
echo -e "${NC}"

print_header "Welcome to MintEDGE Installer"
echo "This script will guide you through installing MintEDGE,"
echo "a flexible edge computing simulation framework."
echo ""
echo "Requirements:"
echo "  - Ubuntu/Debian-based Linux distribution"
echo "  - Python 3.13 (or 3.10+)"
echo "  - sudo privileges for system packages"
echo "  - Internet connection"
echo ""

if ! ask_yes_no "Do you want to proceed with installation?"; then
    echo "Installation cancelled."
    exit 0
fi

# Step 1: Check if running on supported distribution
print_header "Step 1: System Check"

if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $NAME $VERSION"
    
    if [[ ! "$ID" =~ ^(ubuntu|debian|linuxmint|pop)$ ]]; then
        print_warning "This script is optimized for Ubuntu/Debian-based distributions."
        print_warning "Your distribution: $ID"
        if ! ask_yes_no "Do you want to continue anyway?"; then
            exit 1
        fi
    fi
else
    print_warning "Could not detect OS version"
fi

# Step 2: Check Python version
print_header "Step 2: Python Version Check"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    echo "Found Python $PYTHON_VERSION"
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        print_success "Python version is compatible"
        PYTHON_CMD="python3"
    else
        print_error "Python 3.10 or higher is required"
        echo "Current version: $PYTHON_VERSION"
        
        if ask_yes_no "Would you like to see instructions for installing Python 3.13?"; then
            echo ""
            echo "To install Python 3.13 on Ubuntu/Debian:"
            echo "  sudo apt update"
            echo "  sudo apt install software-properties-common -y"
            echo "  sudo add-apt-repository ppa:deadsnakes/ppa -y"
            echo "  sudo apt update"
            echo "  sudo apt install python3.13 python3.13-venv python3.13-dev -y"
            echo ""
        fi
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Step 3: Check for git
print_header "Step 3: Git Check"

if ! command -v git &> /dev/null; then
    print_warning "Git is not installed"
    if ask_yes_no "Do you want to install Git now?"; then
        sudo apt update
        sudo apt install -y git
        print_success "Git installed"
    else
        print_error "Git is required for cloning the repository"
        exit 1
    fi
else
    print_success "Git is already installed"
fi

# Step 4: Installation directory
print_header "Step 4: Installation Directory"

# Function to handle directory selection
select_directory() {
    local default_dir="$HOME/MintEDGE"
    local selected_dir=""
    
    while true; do
        echo "Default installation directory: $default_dir"
        echo ""
        echo "Enter the FULL path where MintEDGE should be installed."
        echo "Example: /home/luka/Documents/dev/MintEDGE"
        echo ""
        read -p "Press Enter to use default or specify full path: " selected_dir
        selected_dir=${selected_dir:-$default_dir}
        selected_dir="${selected_dir%/}"
        
        # Check if path ends with MintEDGE, if not suggest adding it
        if [ -d "$selected_dir" ]; then
            # Directory exists
            if [[ "$selected_dir" != *"/MintEDGE" ]] && [[ "$(basename "$selected_dir")" != "MintEDGE" ]]; then
                print_warning "Directory already exists: $selected_dir"
                echo ""
                echo "It looks like you entered a parent directory."
                echo "MintEDGE will be installed in a subdirectory."
                echo ""
                echo "Options:"
                echo "  1. Install in: $selected_dir/MintEDGE (recommended)"
                echo "  2. Remove existing directory and use: $selected_dir"
                echo "  3. Enter a different path"
                echo "  4. Cancel installation"
                echo ""
                read -p "Enter your choice (1/2/3/4): " choice
                
                case $choice in
                    1)
                        # Use subdirectory
                        INSTALL_DIR="$selected_dir/MintEDGE"
                        if [ -d "$INSTALL_DIR" ]; then
                            print_warning "$INSTALL_DIR already exists"
                            if ask_yes_no "Remove $INSTALL_DIR and proceed?"; then
                                rm -rf "$INSTALL_DIR"
                                print_success "Removed existing directory"
                            else
                                echo "Please choose again."
                                echo ""
                                continue
                            fi
                        fi
                        print_success "Will create directory: $INSTALL_DIR"
                        return 0
                        ;;
                    2)
                        # Remove existing and use it
                        if ask_yes_no "Are you sure you want to REMOVE ALL CONTENTS of $selected_dir?"; then
                            rm -rf "$selected_dir"
                            print_success "Removed existing directory"
                            INSTALL_DIR="$selected_dir"
                            print_success "Will create directory: $INSTALL_DIR"
                            return 0
                        else
                            echo "Removal cancelled. Please choose again."
                            echo ""
                            continue
                        fi
                        ;;
                    3)
                        # Choose different path
                        echo ""
                        continue
                        ;;
                    4)
                        # Cancel
                        print_error "Installation cancelled"
                        exit 1
                        ;;
                    *)
                        echo "Invalid choice. Please enter 1, 2, 3, or 4."
                        echo ""
                        continue
                        ;;
                esac
            else
                print_warning "Directory already exists: $selected_dir"
                echo ""
                echo "Options:"
                echo "  1. Remove existing directory and reinstall"
                echo "  2. Choose a different directory"
                echo "  3. Cancel installation"
                echo ""
                read -p "Enter your choice (1/2/3): " choice
                
                case $choice in
                    1)
                        if ask_yes_no "Are you sure you want to remove $selected_dir?"; then
                            rm -rf "$selected_dir"
                            print_success "Removed existing directory"
                            INSTALL_DIR="$selected_dir"
                            return 0
                        else
                            echo "Removal cancelled. Please choose again."
                            echo ""
                            continue
                        fi
                        ;;
                    2)
                        echo ""
                        continue
                        ;;
                    3)
                        print_error "Installation cancelled"
                        exit 1
                        ;;
                    *)
                        echo "Invalid choice. Please enter 1, 2, or 3."
                        echo ""
                        continue
                        ;;
                esac
            fi
        else
            # Directory doesn't exist!
            INSTALL_DIR="$selected_dir"
            
            # Check if parent directory exists and is writable
            parent_dir=$(dirname "$selected_dir")
            if [ ! -d "$parent_dir" ]; then
                print_error "Parent directory does not exist: $parent_dir"
                echo "Please create it first or choose a different path."
                echo ""
                continue
            fi
            
            if [ ! -w "$parent_dir" ]; then
                print_error "Cannot write to parent directory: $parent_dir"
                echo "Check permissions or choose a different path."
                echo ""
                continue
            fi
            
            print_success "Will create directory: $INSTALL_DIR"
            return 0
        fi
    done
}

# Call the directory selection function
select_directory

# Step 5: Clone repository
print_header "Step 5: Cloning Repository"

echo "Cloning MintEDGE from GitHub..."
git clone https://github.com/blasf1/MintEDGE.git "$INSTALL_DIR"
cd "$INSTALL_DIR"
print_success "Repository cloned successfully"

# Step 6: Virtual environment
print_header "Step 6: Virtual Environment Setup"

echo "Creating Python virtual environment..."
$PYTHON_CMD -m venv .venv

if [ $? -eq 0 ]; then
    print_success "Virtual environment created"
else
    print_error "Failed to create virtual environment"
    print_warning "You may need to install python3-venv:"
    echo "  sudo apt install python3-venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate
print_success "Virtual environment activated"

# Step 7: Install Python dependencies
print_header "Step 7: Installing Python Dependencies"

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing MintEDGE Python dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    exit 1
fi

# Step 8: SUMO installation
print_header "Step 8: SUMO Installation"

if command -v sumo &> /dev/null; then
    SUMO_VERSION=$(sumo --version 2>&1 | head -n1)
    print_success "SUMO is already installed: $SUMO_VERSION"
else
    print_warning "SUMO is not installed"
    echo ""
    echo "SUMO (Simulation of Urban MObility) is required for MintEDGE."
    echo "This will install system packages and may take several minutes."
    echo ""
    
    if ask_yes_no "Do you want to install SUMO now?"; then
        echo "Installing SUMO..."
        sudo apt update
        sudo apt install -y sumo sumo-tools sumo-doc
        
        if command -v sumo &> /dev/null; then
            print_success "SUMO installed successfully"
            sumo --version
        else
            print_error "SUMO installation failed"
            echo ""
            echo "Please install SUMO manually:"
            echo "  sudo apt install sumo sumo-tools sumo-doc"
            exit 1
        fi
    else
        print_warning "Skipping SUMO installation"
        echo "Note: MintEDGE requires SUMO to run"
    fi
fi

# Verify SUMO is in PATH
if ! command -v sumo &> /dev/null; then
    print_warning "SUMO is not in your PATH"
    echo ""
    echo "Common SUMO installation paths:"
    echo "  /usr/bin/sumo"
    echo "  /usr/local/bin/sumo"
    echo ""
    echo "You may need to add SUMO to your PATH. Add this to your ~/.bashrc:"
    echo "  export PATH=\"/usr/share/sumo/bin:\$PATH\""
fi

# Step 9: Verification
print_header "Step 9: Installation Verification"

echo "Running verification checks..."
echo ""

# Check Python imports
echo "Checking Python dependencies..."
$PYTHON_CMD -c "
import sys
failed = []
packages = ['simpy', 'numpy', 'pandas', 'networkx', 'matplotlib', 'libsumo', 'sumolib', 'tqdm', 'requests', 'pyproj', 'multipledispatch']
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        failed.append(pkg)
        
if failed:
    print('Failed to import:', ', '.join(failed))
    sys.exit(1)
else:
    print('All Python dependencies OK')
"

if [ $? -eq 0 ]; then
    print_success "All Python dependencies verified"
else
    print_error "Some Python dependencies are missing"
fi

# Check SUMO
if command -v sumo &> /dev/null; then
    print_success "SUMO is accessible"
else
    print_warning "SUMO is not accessible from command line"
fi

# Step 10: Configuration
print_header "Step 10: Configuration"

echo "The settings.py file contains all simulation parameters."
echo "Location: $INSTALL_DIR/settings.py"
echo ""
echo "Default configuration:"
echo "  - Simulation area: Maastricht, Netherlands"
echo "  - Random routes: Enabled"
echo "  - Number of cars: 2500"
echo "  - Orchestration interval: 60 seconds"
echo ""

if ask_yes_no "Do you want to open settings.py for review/editing?"; then
    ${EDITOR:-nano} settings.py
fi

# Step 11: Test run
print_header "Step 11: Test Run"

echo "You can now test MintEDGE with a short simulation."
echo ""
echo "Example command:"
echo "  python MintEDGE.py --simulation-time 300 --seed 1 --output test_results.csv"
echo ""

if ask_yes_no "Do you want to run a 5-minute test simulation now?"; then
    echo ""
    echo "Starting test simulation (300 seconds = 5 minutes)..."
    echo "This may take several minutes to complete..."
    echo ""
    
    $PYTHON_CMD MintEDGE.py --simulation-time 300 --seed 1 --output test_results.csv
    
    if [ $? -eq 0 ]; then
        print_success "Test simulation completed successfully!"
        if [ -f test_results.csv ]; then
            echo ""
            echo "Results saved to: test_results.csv"
            echo "File size: $(du -h test_results.csv | cut -f1)"
            echo "Number of records: $(wc -l < test_results.csv)"
        fi
    else
        print_error "Test simulation failed"
        echo ""
        echo "Common issues:"
        echo "  1. SUMO not in PATH"
        echo "  2. Missing base station data (bss.csv)"
        echo "  3. Invalid configuration in settings.py"
    fi
else
    echo "Skipping test run"
fi

# Step 12: Completion
print_header "Installation Complete!"

echo "MintEDGE has been installed to: $INSTALL_DIR"
echo ""
echo "To use MintEDGE:"
echo "  1. cd $INSTALL_DIR"
echo "  2. source .venv/bin/activate"
echo "  3. python MintEDGE.py --simulation-time <seconds> --seed <seed> --output <file.csv>"
echo ""
echo "Quick start command:"
echo "  cd $INSTALL_DIR && source .venv/bin/activate"
echo ""
echo "Configuration file:"
echo "  $INSTALL_DIR/settings.py"
echo ""
echo "Documentation:"
echo "  https://github.com/blasf1/MintEDGE"
echo ""

if [ -f test_results.csv ]; then
    echo "Test results:"
    echo "  $INSTALL_DIR/test_results.csv"
    echo ""
fi

print_success "Installation successful! Happy simulating! 🚀"
echo ""

# Create activation helper script
cat > activate.sh << 'EOF'
#!/bin/bash
# Quick activation script for MintEDGE virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/.venv/bin/activate"
echo "MintEDGE virtual environment activated"
echo "Run: python MintEDGE.py --help"
EOF
chmod +x activate.sh

echo "Tip: Use './activate.sh' to quickly activate the virtual environment"
echo ""
