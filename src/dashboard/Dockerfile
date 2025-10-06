FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY src/dashboard/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/dashboard/src ./src
COPY src/dashboard/public ./public
COPY src/dashboard/tsconfig.json ./

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY infrastructure/nginx.conf /etc/nginx/conf.d/default.conf

# Create nginx user directory
RUN mkdir -p /var/cache/nginx/client_temp \
             /var/cache/nginx/proxy_temp \
             /var/cache/nginx/fastcgi_temp \
             /var/cache/nginx/uwsgi_temp \
             /var/cache/nginx/scgi_temp

# Set proper permissions
RUN chown -R nginx:nginx /var/cache/nginx \
    && chown -R nginx:nginx /usr/share/nginx/html \
    && chmod -R 755 /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
