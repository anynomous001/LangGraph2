# Complete Docker Guide: From Basics to Advanced

## Table of Contents
1. [Introduction to Docker](#introduction-to-docker)
2. [Docker Architecture](#docker-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Docker Basics](#docker-basics)
5. [Working with Images](#working-with-images)
6. [Container Management](#container-management)
7. [Dockerfile Deep Dive](#dockerfile-deep-dive)
8. [Docker Networking](#docker-networking)
9. [Data Management and Volumes](#data-management-and-volumes)
10. [Docker Compose](#docker-compose)
11. [Multi-Stage Builds](#multi-stage-builds)
12. [Docker Registry](#docker-registry)
13. [Security Best Practices](#security-best-practices)
14. [Monitoring and Logging](#monitoring-and-logging)
15. [Production Deployment](#production-deployment)
16. [Advanced Topics](#advanced-topics)
17. [Real-World Use Cases](#real-world-use-cases)
18. [Troubleshooting](#troubleshooting)
19. [Recommended Resources](#recommended-resources)

---

## 1. Introduction to Docker

### What is Docker?
Docker is a containerization platform that enables developers to package applications and their dependencies into lightweight, portable containers. These containers can run consistently across different environments, from development laptops to production servers.

### Key Benefits
- **Consistency**: "Works on my machine" becomes "works everywhere"
- **Portability**: Run anywhere Docker is installed
- **Efficiency**: Share OS kernel, lighter than VMs
- **Scalability**: Easy horizontal scaling
- **Isolation**: Secure process and resource isolation
- **Speed**: Fast startup times and deployment

### Docker vs Virtual Machines
```
Virtual Machines:
[App A] [App B] [App C]
[Guest OS] [Guest OS] [Guest OS]
[Hypervisor]
[Host OS]
[Infrastructure]

Docker Containers:
[App A] [App B] [App C]
[Docker Engine]
[Host OS]
[Infrastructure]
```

---

## 2. Docker Architecture

### Core Components

#### Docker Engine
The runtime that manages containers, images, networks, and volumes.

#### Docker Client
Command-line interface (CLI) that communicates with Docker daemon.

#### Docker Daemon
Background service managing Docker objects.

#### Docker Objects
- **Images**: Read-only templates for creating containers
- **Containers**: Runnable instances of images
- **Networks**: Enable container communication
- **Volumes**: Persistent data storage

### Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐
│   Docker CLI    │────│  Docker Daemon  │
└─────────────────┘    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌─────────┐         ┌─────────────┐       ┌─────────────┐
   │ Images  │         │ Containers  │       │  Networks   │
   └─────────┘         └─────────────┘       └─────────────┘
        │                     │                     │
   ┌─────────┐         ┌─────────────┐       ┌─────────────┐
   │Registry │         │   Volumes   │       │   Plugins   │
   └─────────┘         └─────────────┘       └─────────────┘
```

---

## 3. Installation and Setup

### Linux Installation
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up repository
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Add user to docker group
sudo usermod -aG docker $USER
```

### Verification
```bash
# Check Docker version
docker --version

# Run hello world
docker run hello-world

# Check system info
docker system info
```

---

## 4. Docker Basics

### Essential Commands

#### Image Commands
```bash
# Pull an image
docker pull nginx:latest

# List images
docker images

# Remove image
docker rmi nginx:latest

# Search for images
docker search ubuntu
```

#### Container Commands
```bash
# Run a container
docker run nginx

# Run container in background
docker run -d nginx

# Run with port mapping
docker run -d -p 8080:80 nginx

# Run interactively
docker run -it ubuntu /bin/bash

# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>
```

---

## 5. Working with Images

### Understanding Image Layers
Images are built in layers using a Union File System. Each instruction in a Dockerfile creates a new layer.

```dockerfile
FROM ubuntu:20.04          # Layer 1: Base OS
RUN apt-get update         # Layer 2: Package update
RUN apt-get install -y nginx  # Layer 3: Install nginx
COPY index.html /var/www/html/  # Layer 4: Copy files
CMD ["nginx", "-g", "daemon off;"]  # Layer 5: Default command
```

### Image Management
```bash
# Inspect image details
docker inspect nginx:latest

# View image history
docker history nginx:latest

# Tag an image
docker tag nginx:latest my-nginx:v1.0

# Save image to file
docker save -o nginx.tar nginx:latest

# Load image from file
docker load -i nginx.tar

# Remove unused images
docker image prune
```

---

## 6. Container Management

### Container Lifecycle
```bash
# Create container without starting
docker create --name my-container nginx

# Start existing container
docker start my-container

# Restart container
docker restart my-container

# Pause/unpause container
docker pause my-container
docker unpause my-container

# Stop container gracefully
docker stop my-container

# Force stop container
docker kill my-container

# Remove stopped container
docker rm my-container
```

### Container Interaction
```bash
# Execute command in running container
docker exec -it my-container /bin/bash

# Copy files to/from container
docker cp file.txt my-container:/app/
docker cp my-container:/app/logs.txt ./

# View container logs
docker logs my-container
docker logs -f my-container  # Follow logs

# Monitor container stats
docker stats my-container
```

---

## 7. Dockerfile Deep Dive

### Dockerfile Instructions

#### FROM
```dockerfile
# Use specific version
FROM node:16-alpine

# Multi-stage build
FROM node:16-alpine AS build
FROM nginx:alpine AS production
```

#### RUN
```dockerfile
# Single command
RUN npm install

# Multiple commands (better caching)
RUN apt-get update && \
    apt-get install -y \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*
```

#### COPY vs ADD
```dockerfile
# COPY (preferred for simple file copying)
COPY package.json /app/
COPY src/ /app/src/

# ADD (for URLs and archives)
ADD https://example.com/file.tar.gz /app/
ADD archive.tar.gz /app/  # Auto-extracts
```

#### WORKDIR
```dockerfile
WORKDIR /app
# Equivalent to: RUN cd /app
```

#### ENV
```dockerfile
ENV NODE_ENV=production
ENV API_URL=https://api.example.com
ENV PORT=3000
```

#### EXPOSE
```dockerfile
EXPOSE 3000
EXPOSE 80 443
```

#### USER
```dockerfile
# Create and use non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

#### CMD vs ENTRYPOINT
```dockerfile
# CMD (can be overridden)
CMD ["npm", "start"]

# ENTRYPOINT (always runs)
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["npm", "start"]

# Combined usage
ENTRYPOINT ["python"]
CMD ["app.py"]
```

### Real-World Dockerfile Examples

#### Node.js Application
```dockerfile
FROM node:16-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM node:16-alpine

# Create app user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

WORKDIR /app

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
```

#### Python Flask Application
```dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

---

## 8. Docker Networking

### Network Types

#### Bridge (Default)
```bash
# Create custom bridge network
docker network create --driver bridge my-network

# Run container on custom network
docker run -d --name web --network my-network nginx
```

#### Host
```bash
# Use host networking
docker run -d --network host nginx
```

#### None
```bash
# No networking
docker run -d --network none alpine sleep 3600
```

#### Overlay (Swarm Mode)
```bash
# Create overlay network
docker network create --driver overlay --attachable my-overlay
```

### Network Management
```bash
# List networks
docker network ls

# Inspect network
docker network inspect bridge

# Connect container to network
docker network connect my-network my-container

# Disconnect container from network
docker network disconnect my-network my-container

# Remove network
docker network rm my-network
```

### Real-World Example: Multi-Container Application
```bash
# Create network
docker network create app-network

# Run database
docker run -d \
  --name postgres-db \
  --network app-network \
  -e POSTGRES_DB=myapp \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  postgres:13

# Run application
docker run -d \
  --name web-app \
  --network app-network \
  -p 3000:3000 \
  -e DATABASE_URL=postgresql://user:password@postgres-db:5432/myapp \
  my-web-app:latest
```

---

## 9. Data Management and Volumes

### Volume Types

#### Named Volumes
```bash
# Create named volume
docker volume create my-data

# Use named volume
docker run -d -v my-data:/data nginx

# List volumes
docker volume ls

# Inspect volume
docker volume inspect my-data
```

#### Bind Mounts
```bash
# Mount host directory
docker run -d -v /host/path:/container/path nginx

# Mount current directory
docker run -d -v $(pwd):/app node:16 npm start
```

#### tmpfs Mounts
```bash
# Mount tmpfs (memory-based)
docker run -d --tmpfs /tmp nginx
```

### Volume Management
```bash
# Remove unused volumes
docker volume prune

# Remove specific volume
docker volume rm my-data

# Backup volume data
docker run --rm -v my-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/backup.tar.gz -C /data .

# Restore volume data
docker run --rm -v my-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/backup.tar.gz -C /data
```

### Database Persistence Example
```bash
# Run PostgreSQL with persistent data
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=mypassword \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:13
```

---

## 10. Docker Compose

### Introduction to Compose
Docker Compose is a tool for defining and running multi-container applications using a YAML file.

### Basic docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
      - /app/node_modules

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:

networks:
  default:
    driver: bridge
```

### Advanced Compose Features

#### Environment Variables
```yaml
version: '3.8'

services:
  web:
    image: my-app:${TAG:-latest}
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - API_KEY=${API_KEY}
    env_file:
      - .env
      - .env.local
```

#### Healthchecks
```yaml
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

#### Scaling Services
```yaml
services:
  web:
    image: my-app
    deploy:
      replicas: 3
    ports:
      - "3000-3002:3000"
```

### Compose Commands
```bash
# Start services
docker-compose up
docker-compose up -d  # Background

# Build and start
docker-compose up --build

# Scale service
docker-compose up --scale web=3

# Stop services
docker-compose down

# View logs
docker-compose logs
docker-compose logs -f web

# Execute commands
docker-compose exec web bash

# Restart service
docker-compose restart web
```

### Real-World Example: LAMP Stack
```yaml
version: '3.8'

services:
  php:
    build:
      context: .
      dockerfile: Dockerfile.php
    volumes:
      - ./src:/var/www/html
    depends_on:
      - mysql

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./src:/var/www/html
      - ./ssl:/etc/ssl
    depends_on:
      - php

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: myapp
      MYSQL_USER: appuser
      MYSQL_PASSWORD: apppassword
    volumes:
      - mysql_data:/var/lib/mysql
    ports:
      - "3306:3306"

  phpmyadmin:
    image: phpmyadmin:latest
    environment:
      PMA_HOST: mysql
      PMA_PORT: 3306
    ports:
      - "8080:80"
    depends_on:
      - mysql

volumes:
  mysql_data:

networks:
  default:
    driver: bridge
```

---

## 11. Multi-Stage Builds

### Benefits
- Smaller final images
- Separate build and runtime environments
- Better security (no build tools in production)
- Improved build caching

### Example: Node.js Application
```dockerfile
# Build stage
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force
COPY . .
RUN npm run build

# Test stage
FROM builder AS tester
RUN npm ci
RUN npm test

# Production stage
FROM node:16-alpine AS production
WORKDIR /app

# Copy only production dependencies and built assets
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001
USER nextjs

EXPOSE 3000
CMD ["npm", "start"]
```

### Example: Go Application
```dockerfile
# Build stage
FROM golang:1.19-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

# Production stage
FROM alpine:latest AS production
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .
CMD ["./main"]
```

---

## 12. Docker Registry

### Docker Hub
```bash
# Login to Docker Hub
docker login

# Tag image for registry
docker tag my-app:latest username/my-app:latest

# Push to registry
docker push username/my-app:latest

# Pull from registry
docker pull username/my-app:latest
```

### Private Registry
```bash
# Run local registry
docker run -d -p 5000:5000 --name registry registry:2

# Tag for local registry
docker tag my-app:latest localhost:5000/my-app:latest

# Push to local registry
docker push localhost:5000/my-app:latest
```

### Registry with Authentication
```yaml
version: '3.8'

services:
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    environment:
      REGISTRY_AUTH: htpasswd
      REGISTRY_AUTH_HTPASSWD_REALM: Registry Realm
      REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
      REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY: /data
    volumes:
      - registry_data:/data
      - ./auth:/auth

volumes:
  registry_data:
```

---

## 13. Security Best Practices

### Image Security
```dockerfile
# Use official base images
FROM node:16-alpine

# Use specific versions
FROM node:16.17.0-alpine3.16

# Use non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001
USER nextjs

# Minimize attack surface
RUN apk add --no-cache curl && \
    rm -rf /var/cache/apk/*

# Use multi-stage builds
FROM node:16-alpine AS builder
# ... build steps
FROM node:16-alpine AS production
# ... minimal production image
```

### Runtime Security
```bash
# Run with limited privileges
docker run --user 1000:1000 my-app

# Limit resources
docker run --memory="512m" --cpus="0.5" my-app

# Use security options
docker run --security-opt no-new-privileges my-app

# Read-only filesystem
docker run --read-only my-app

# Drop capabilities
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE my-app
```

### Security Scanning
```bash
# Scan image for vulnerabilities
docker scan my-app:latest

# Use security tools
# Trivy
trivy image my-app:latest

# Clair
# Anchore
```

---

## 14. Monitoring and Logging

### Container Monitoring
```bash
# Real-time stats
docker stats

# Resource usage
docker system df
docker system events

# Container processes
docker top <container>
```

### Logging
```bash
# View logs
docker logs <container>
docker logs -f <container>  # Follow
docker logs --since=1h <container>  # Time filter
docker logs --tail=100 <container>  # Limit lines
```

### Monitoring Stack with Compose
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

  node-exporter:
    image: prom/node-exporter
    ports:
      - "9100:9100"

  cadvisor:
    image: gcr.io/cadvisor/cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro

volumes:
  prometheus_data:
  grafana_data:
```

---

## 15. Production Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Add worker nodes
docker swarm join --token <token> <manager-ip>:2377

# Deploy stack
docker stack deploy -c docker-compose.yml myapp

# Scale services
docker service scale myapp_web=5

# Update service
docker service update --image myapp:v2 myapp_web
```

### Kubernetes Integration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: my-app:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
---
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

### CI/CD Pipeline Example (GitHub Actions)
```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ${{ secrets.REGISTRY_URL }}/myapp:${{ github.sha }} .
        docker build -t ${{ secrets.REGISTRY_URL }}/myapp:latest .
    
    - name: Login to registry
      run: echo "${{ secrets.REGISTRY_PASSWORD }}" | docker login ${{ secrets.REGISTRY_URL }} -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
    
    - name: Push images
      run: |
        docker push ${{ secrets.REGISTRY_URL }}/myapp:${{ github.sha }}
        docker push ${{ secrets.REGISTRY_URL }}/myapp:latest
    
    - name: Deploy to production
      run: |
        # Deploy using Docker Swarm, Kubernetes, or other orchestrator
        kubectl set image deployment/myapp myapp=${{ secrets.REGISTRY_URL }}/myapp:${{ github.sha }}
```

---

## 16. Advanced Topics

### Docker BuildKit
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Advanced Dockerfile with BuildKit
# syntax=docker/dockerfile:1
FROM node:16-alpine

# Mount cache for npm
RUN --mount=type=cache,target=/root/.npm \
    npm install

# Mount secret
RUN --mount=type=secret,id=api_key \
    API_KEY=$(cat /run/secrets/api_key) npm run build
```

### Custom Networks
```bash
# Create macvlan network
docker network create -d macvlan \
  --subnet=192.168.1.0/24 \
  --gateway=192.168.1.1 \
  -o parent=eth0 \
  macvlan-net
```

### Container Orchestration Patterns

#### Service Discovery
```yaml
version: '3.8'

services:
  consul:
    image: consul:1.6
    command: consul agent -server -ui -node=server-1 -bootstrap-expect=1 -client=0.0.0.0
    ports:
      - "8500:8500"

  registrator:
    image: gliderlabs/registrator
    command: -internal consul://consul:8500
    volumes:
      - /var/run/docker.sock:/tmp/docker.sock
    depends_on:
      - consul
```

#### Load Balancing
```yaml
version: '3.8'

services:
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app1
      - app2
      - app3

  app1:
    image: my-app
    expose:
      - "3000"

  app2:
    image: my-app
    expose:
      - "3000"

  app3:
    image: my-app
    expose:
      - "3000"
```

---

## 17. Real-World Use Cases

### Microservices Architecture
```yaml
version: '3.8'

services:
  api-gateway:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - user-service
      - product-service
      - order-service

  user-service:
    build: ./user-service
    environment:
      - DATABASE_URL=postgresql://user:pass@user-db:5432/users
    depends_on:
      - user-db

  product-service:
    build: ./product-service
    environment:
      - DATABASE_URL=postgresql://user:pass@product-db:5432/products
    depends_on:
      - product-db

  order-service:
    build: ./order-service
    environment:
      - DATABASE_URL=postgresql://user:pass@order-db:5432/orders
      - USER_SERVICE_URL=http://user-service:3000
      - PRODUCT_SERVICE_URL=http://product-service:3000
    depends_on:
      - order-db

  user-db:
    image: postgres:13
    environment:
      POSTGRES_DB: users
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - user_data:/var/lib/postgresql/data

  product-db:
    image: postgres:13
    environment:
      POSTGRES_DB: products
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - product_data:/var/lib/postgresql/data

  order-db:
    image: postgres:13
    environment:
      POSTGRES_DB: orders
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - order_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

volumes:
  user_data:
  product_data:
  order_data:
  redis_data:
  rabbitmq_data:
```

### Development Environment
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - CHOKIDAR_USEPOLLING=true
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: myapp_dev
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  mailhog:
    image: mailhog/mailhog
    ports:
      - "1025:1025"
      - "8025:8025"

volumes:
  postgres_dev:
```

---

## 18. Troubleshooting

### Common Issues and Solutions

#### Container Won't Start
```bash
# Check container logs
docker logs <container>

# Check container configuration
docker inspect <container>

# Run container interactively
docker run -it <image> /bin/bash
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./data

# Use correct user in Dockerfile
USER 1000:1000
```

#### Network Connectivity
```bash
# Test connectivity between containers
docker exec -it container1 ping container2

# Check network configuration
docker network inspect <network>

# List container ports
docker port <container>
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check system resources
docker system df
docker system prune

# Analyze image layers
docker history <image>
```

#### Build Issues
```bash
# Clear build cache
docker builder prune

# Build with no cache
docker build --no-cache .

# Debug build process
docker build --progress=plain .
```

---

## 19. Recommended Resources

### Essential Docker Blogs and Resources

#### Official Documentation
- **Docker Official Documentation**: https://docs.docker.com/
  - Comprehensive official documentation with tutorials, reference guides, and best practices

#### Industry-Leading Blogs
- **Docker Blog**: https://www.docker.com/blog/
  - Official Docker company blog with latest updates, tutorials, and case studies

- **Kubernetes Blog**: https://kubernetes.io/blog/
  - Official Kubernetes blog covering container orchestration and Docker integration

- **Container Journal**: https://containerjournal.com/
  - Industry news, trends, and technical articles about containerization

- **The New Stack**: https://thenewstack.io/
  - In-depth coverage of cloud-native technologies, Docker, and container ecosystem

#### Technical Deep-Dive Blogs
- **Red Hat Developer**: https://developers.redhat.com/topics/containers
  - Enterprise-focused container tutorials and best practices

- **NGINX Blog**: https://www.nginx.com/blog/
  - Load balancing, reverse proxy, and microservices architecture with Docker

- **Traefik Blog**: https://traefik.io/blog/
  - Modern reverse proxy and load balancer for containers

#### Community and Learning Resources
- **Docker Captain's Blog**: Various Docker Captains maintain excellent blogs
  - Bret Fisher: https://www.bretfisher.com/
  - Nigel Poulton: https://nigelpoulton.com/
  - Adrian Mouat: https://container-solutions.com/author/adrian/

- **Medium - Docker Tag**: https://medium.com/tag/docker
  - Community-driven articles and tutorials

- **Dev.to Docker Community**: https://dev.to/t/docker
  - Developer community sharing Docker tips and experiences

#### Video Learning Resources
- **Docker's Official YouTube Channel**: https://www.youtube.com/user/dockerrun
  - Official tutorials, webinars, and conference talks

- **TechWorld with Nana**: https://www.youtube.com/c/TechWorldwithNana
  - Excellent DevOps and Docker tutorials

#### Books and Advanced Resources
- **"Docker Deep Dive"** by Nigel Poulton
- **"Docker in Practice"** by Ian Miell and Aidan Hobson Sayers
- **"Kubernetes in Action"** by Marko Lukša
- **"Building Microservices"** by Sam Newman

#### Tools and Utilities
- **Dive**: Analyze Docker image layers
  - https://github.com/wagoodman/dive

- **Hadolint**: Dockerfile linter
  - https://github.com/hadolint/hadolint

- **Docker Bench Security**: Security scanner
  - https://github.com/docker/docker-bench-security

#### Online Courses and Certifications
- **Docker Certified Associate (DCA)**: Official Docker certification
- **Linux Academy/A Cloud Guru**: Docker and Kubernetes courses
- **Udemy**: Various Docker courses by instructors like Bret Fisher

---

## Conclusion

This comprehensive guide covers Docker from fundamental concepts to advanced production deployment scenarios. The key to mastering Docker is hands-on practice with real-world applications.

### Next Steps for Learning
1. **Start Small**: Begin with simple applications and gradually increase complexity
2. **Practice Regularly**: Set up development environments using Docker
3. **Explore Orchestration**: Learn Kubernetes or Docker Swarm for production deployments
4. **Security Focus**: Always implement security best practices
5. **Community Engagement**: Join Docker communities and contribute to open-source projects

### Key Takeaways
- **Consistency**: Docker ensures applications run the same everywhere
- **Efficiency**: Containers are lightweight and resource-efficient
- **Scalability**: Easy to scale applications horizontally
- **DevOps Integration**: Essential tool for modern CI/CD pipelines
- **Microservices**: Enables microservices architecture patterns

### Production Readiness Checklist
- [ ] Multi-stage builds implemented
- [ ] Non-root user configured
- [ ] Health checks defined
- [ ] Resource limits set
- [ ] Secrets management configured
- [ ] Logging strategy implemented
- [ ] Monitoring and alerting setup
- [ ] Security scanning integrated
- [ ] Backup and disaster recovery planned
- [ ] Documentation maintained

Remember: Docker is a tool that enables better software delivery, but it's the practices and patterns around it that create real value in production environments. Focus on understanding not just how to use Docker, but when and why to use specific features and patterns.

---

*This guide serves as a comprehensive reference for Docker usage. Keep it handy and refer back to specific sections as needed. The containerization ecosystem evolves rapidly, so stay updated with the latest developments through the recommended resources.*
