# 🚀 AffectRON Deployment Guide

AffectRON can be deployed in multiple ways depending on your needs:

## 📋 Prerequisites

- Docker and Docker Compose
- Git
- At least 4GB RAM available
- Ports 5432, 6379, 8000, 3000 available (or configure in .env)

## 🏠 Local Development Deployment

### Quick Start (Linux/Mac)

```bash
# Clone repository
git clone https://github.com/Gzeu/AffectRON.git
cd AffectRON

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Deploy locally
./deploy.sh local

# Or manually with Docker Compose
docker-compose up -d
```

### Quick Start (Windows PowerShell)

```powershell
# Clone repository
git clone https://github.com/Gzeu/AffectRON.git
cd AffectRON

# Copy environment file
Copy-Item .env.example .env
# Edit .env with your configuration

# Deploy locally
docker-compose up -d
```

## 🏭 Production Deployment

### Option 1: Enhanced Docker Compose

```bash
# Deploy with production profile (includes Nginx)
./deploy.sh production

# Or manually
docker-compose --profile production up -d
```

This includes:
- Load balancer (Nginx)
- SSL termination
- Monitoring stack (Prometheus + Grafana)
- ELK stack for logging (optional)

### Option 2: Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
./deploy.sh k8s
```

**Prerequisites for Kubernetes:**
- kubectl configured
- cert-manager installed for SSL certificates
- NGINX Ingress Controller

## 🧪 Testing

```bash
# Run test suite
./deploy.sh test
```

## 🛠️ Services Overview

| Service | Port | Description |
|---------|------|-------------|
| **API Backend** | 8000 | FastAPI REST API |
| **Dashboard** | 3000 | React web interface |
| **PostgreSQL** | 5432 | Primary database |
| **Redis** | 6379 | Cache & message broker |
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3001 | Dashboards & visualization |
| **Nginx** | 80/443 | Load balancer (production) |

## 🔧 Configuration

### Environment Variables

Key variables in `.env`:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://affectron:password@postgres:5432/affectron

# API Security
SECRET_KEY=your-32-character-secret-key

# External APIs (optional)
TWITTER_BEARER_TOKEN=your_twitter_token
NEWS_API_KEY=your_news_api_key

# Monitoring
SENTRY_DSN=your_sentry_dsn
```

### Production Checklist

- [ ] Change all default passwords
- [ ] Set strong SECRET_KEY (32+ characters)
- [ ] Configure SSL certificates
- [ ] Set up monitoring alerts
- [ ] Configure backup strategy
- [ ] Set up log aggregation
- [ ] Configure domain name

## 🔍 Monitoring & Health Checks

### Health Endpoints

- **API Health**: `http://localhost:8000/health`
- **Database**: PostgreSQL health checks configured
- **Redis**: Built-in health checks

### Monitoring Dashboards

- **Grafana**: `http://localhost:3001` (admin/grafana_password)
- **Prometheus**: `http://localhost:9090`

### Logs

- Application logs: `./logs/` directory
- Docker logs: `docker-compose logs [service-name]`
- ELK Stack: Kibana at `http://localhost:5601` (optional)

## 🚨 Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Check what's using ports
netstat -tulpn | grep :PORT
# Or change ports in .env file
```

**Memory issues:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
```

**Permission denied:**
```bash
# Fix script permissions (Linux/Mac)
chmod +x deploy.sh
```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs dashboard

# Follow logs in real-time
docker-compose logs -f api
```

## 🔄 Updates and Maintenance

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and deploy
./deploy.sh local
```

### Backup Database

```bash
# Create database backup
docker-compose exec postgres pg_dump -U affectron affectron > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Scale Services

```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale dashboard
docker-compose up -d --scale dashboard=2
```

## 📞 Support

For issues and questions:
1. Check the logs: `docker-compose logs`
2. Verify environment configuration
3. Check service health: `curl http://localhost:8000/health`
4. Review the troubleshooting section above

## 🔒 Security Considerations

- Change all default passwords before production
- Use strong SECRET_KEY (32+ characters)
- Configure proper SSL certificates
- Set up firewall rules
- Regular security updates
- Monitor access logs

---

**AffectRON** - Professional Financial Sentiment Analysis for Romanian Markets 🇷🇴
