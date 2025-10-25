#!/bin/sh

# Check for development mode
if [ "$DISABLE_SSL" = "true" ]; then
  echo "Starting Nginx in development mode (HTTP only)..."
  
  # Create a basic Nginx config for development
  cat > /etc/nginx/conf.d/default.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-XSS-Protection "1; mode=block";
    add_header X-Content-Type-Options "nosniff";
}
EOF

  # Start Nginx
  nginx -g 'daemon off;'
  exit 0
fi

# Production mode continues below with SSL setup

# Check if domain name is provided
if [ -z "$DOMAIN" ]; then
    echo "Error: DOMAIN environment variable is not set"
    exit 1
fi

# Check if email is provided
if [ -z "$EMAIL" ]; then
    echo "Error: EMAIL environment variable is not set"
    exit 1
fi

# Install necessary tools
apk add --no-cache psmisc net-tools curl

# Create required directories
mkdir -p /var/log/nginx
mkdir -p /etc/nginx/ssl

# Start nginx in the background (HTTP only at this point)
echo "Starting Nginx on port 80..."
nginx

# Wait and verify nginx is running
for i in $(seq 1 5); do
    if pgrep nginx > /dev/null; then
        echo "Nginx started successfully"
        break
    fi
    if [ $i -eq 5 ]; then
        echo "Failed to start Nginx after 5 attempts"
        exit 1
    fi
    echo "Waiting for Nginx to start... (attempt $i)"
    sleep 2
done

# Verify Nginx is responding
echo "Verifying Nginx is responding..."
if curl -s -I http://localhost:80 > /dev/null; then
    echo "Nginx is responding on port 80"
else
    echo "Error: Nginx is not responding on port 80"
    exit 1
fi

# Get SSL certificates using standalone mode
echo "Stopping nginx temporarily for certificate acquisition..."
nginx -s stop
sleep 2

echo "Obtaining SSL certificates using standalone mode..."
certbot certonly \
    --standalone \
    --non-interactive \
    --expand \
    --agree-tos \
    --email $EMAIL \
    --domains "$DOMAIN,www.$DOMAIN" \
    --preferred-challenges http \
    --http-01-port 80 \
    --rsa-key-size 4096

# Update nginx configuration with SSL settings
echo "Configuring SSL in Nginx..."
cat > /etc/nginx/conf.d/default.conf <<EOF
# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN www.$DOMAIN;
    return 301 https://\$host\$request_uri;
}

# HTTPS configuration
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    
    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
        add_header Cache-Control "no-cache";
    }

    location /assets {
        expires 1y;
        add_header Cache-Control "public, no-transform";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
EOF

# Start nginx with the new configuration
echo "Starting Nginx with SSL configuration..."
nginx -g "daemon off;"