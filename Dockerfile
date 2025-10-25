# Base stage for building the static files
FROM node:lts AS base
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Runtime stage for serving the application
FROM nginx:mainline-alpine-slim AS runtime

# Install certbot and its nginx plugin
RUN apk add --no-cache certbot certbot-nginx

# Create SSL directory
RUN mkdir -p /etc/nginx/ssl

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files
COPY --from=base ./app/dist /usr/share/nginx/html

# Expose both HTTP and HTTPS ports
EXPOSE 80 443

# Add script to handle SSL certificate
COPY init-ssl.sh /docker-entrypoint.d/
RUN chmod +x /docker-entrypoint.d/init-ssl.sh

CMD ["nginx", "-g", "daemon off;"]