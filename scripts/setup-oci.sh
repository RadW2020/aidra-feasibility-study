#!/bin/bash
# ==============================================================================
# AIDRA - OCI ARM A1 Provisioning Script
# Prerequisito: cuenta OCI Free Tier con instancia ARM A1 (4 OCPU, 24 GB)
# Compatible con Oracle Linux 8 y Ubuntu 22.04
# ==============================================================================
set -euo pipefail

echo "========================================="
echo "AIDRA — OCI ARM A1 Provisioning"
echo "========================================="

# Detect OS
if [ -f /etc/oracle-release ] || [ -f /etc/redhat-release ]; then
    OS="oracle"
    echo "[INFO] Detected Oracle Linux / RHEL-based system"
elif [ -f /etc/lsb-release ] || grep -qi ubuntu /etc/os-release 2>/dev/null; then
    OS="ubuntu"
    echo "[INFO] Detected Ubuntu system"
else
    echo "[WARN] Unknown OS. Attempting Oracle Linux commands."
    OS="oracle"
fi

# ==============================================================================
# 1. Actualizar sistema
# ==============================================================================
echo ""
echo "[STEP 1/6] Updating system packages..."
if [ "$OS" = "oracle" ]; then
    sudo dnf update -y
else
    sudo apt update && sudo apt upgrade -y
fi

# ==============================================================================
# 2. Instalar Docker
# ==============================================================================
echo ""
echo "[STEP 2/6] Installing Docker..."
if command -v docker &>/dev/null; then
    echo "[INFO] Docker already installed: $(docker --version)"
else
    if [ "$OS" = "oracle" ]; then
        sudo dnf install -y dnf-utils
        sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    else
        sudo apt install -y ca-certificates curl gnupg lsb-release
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt update
        sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    fi
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    echo "[INFO] Docker installed: $(docker --version)"
    echo "[INFO] You may need to log out and back in for group changes to take effect."
fi

# ==============================================================================
# 3. Configurar firewall
# ==============================================================================
echo ""
echo "[STEP 3/6] Configuring firewall..."
if [ "$OS" = "oracle" ]; then
    if command -v firewall-cmd &>/dev/null; then
        sudo firewall-cmd --permanent --add-port=8000/tcp  # API FastAPI
        sudo firewall-cmd --permanent --add-port=3000/tcp  # Grafana
        sudo firewall-cmd --reload
        echo "[INFO] Firewall configured (ports 8000, 3000 opened)"
    else
        echo "[WARN] firewall-cmd not found, skipping firewall config"
    fi
else
    if command -v ufw &>/dev/null; then
        sudo ufw allow 8000/tcp comment 'AIDRA API'
        sudo ufw allow 3000/tcp comment 'AIDRA Grafana'
        echo "[INFO] UFW configured (ports 8000, 3000 opened)"
    else
        echo "[WARN] ufw not found, skipping firewall config"
    fi
fi

# ==============================================================================
# 4. Crear directorios de datos
# ==============================================================================
echo ""
echo "[STEP 4/6] Creating data directories..."
sudo mkdir -p /opt/aidra/{models,images,data}
sudo chown -R "$USER":"$USER" /opt/aidra
echo "[INFO] Directories created:"
ls -la /opt/aidra/

# ==============================================================================
# 5. Configurar swap (seguridad para picos de RAM)
# ==============================================================================
echo ""
echo "[STEP 5/6] Configuring 4 GB swap..."
if [ -f /swapfile ]; then
    echo "[INFO] Swap file already exists, skipping"
    swapon --show
else
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
    echo "[INFO] 4 GB swap configured"
    swapon --show
fi

# ==============================================================================
# 6. Limites de open files para PostgreSQL
# ==============================================================================
echo ""
echo "[STEP 6/6] Configuring system limits..."
if ! grep -q 'fs.file-max = 65536' /etc/sysctl.conf 2>/dev/null; then
    echo 'fs.file-max = 65536' | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
    echo "[INFO] fs.file-max set to 65536"
else
    echo "[INFO] fs.file-max already configured"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "========================================="
echo "AIDRA OCI provisioning complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Clone the AIDRA repo:    git clone <repo-url> AIDRA && cd AIDRA"
echo "  2. Configure env vars:      cp .env.example .env && nano .env"
echo "  3. Download models:         ./scripts/download-models.sh"
echo "  4. Build and start:         docker compose build && docker compose up -d"
echo "  5. Verify:                  curl http://localhost:8000/api/health"
echo ""
echo "Services:"
echo "  API + Swagger:  http://<your-ip>:8000/docs"
echo "  Grafana:        http://<your-ip>:3000"
echo "  Prometheus:     http://<your-ip>:9090 (internal only recommended)"
echo ""
