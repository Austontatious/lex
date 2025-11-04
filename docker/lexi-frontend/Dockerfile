# syntax=docker/dockerfile:1.7

FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm config set registry https://registry.npmjs.org/ \
 && npm cache clean --force
RUN npm ci --no-audit --legacy-peer-deps
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY docker/lexi-frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/lexi-frontend/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
COPY --from=build /app/build /usr/share/nginx/html
ENV LEX_API_BASE=http://localhost:9000/lexi \
    COMFY_URL=http://172.17.0.1:8188 \
    VLLM_URL=http://172.17.0.1:8008/v1
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
