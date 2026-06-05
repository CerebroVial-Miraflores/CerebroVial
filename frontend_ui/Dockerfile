# Etapa 1: Construcción
FROM node:20-alpine AS build
WORKDIR /app

# Copiar archivos de dependencias
COPY package*.json ./
RUN npm install

# Copiar el resto del código y construir
COPY . .

# VITE_* inyectables por build-arg. Default a localhost (deploy local actual);
# un deploy con otro host solo cambia los build-args, sin tocar código. Vite
# hornea estas envs en el bundle en build-time (import.meta.env).
ARG VITE_CORE_API_URL=http://localhost:8001
ARG VITE_EDGE_API_URL=http://localhost:8000
ENV VITE_CORE_API_URL=$VITE_CORE_API_URL
ENV VITE_EDGE_API_URL=$VITE_EDGE_API_URL

RUN npm run build

# Etapa 2: Servidor de producción (ligero)
FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html

# Reemplaza el default.conf del nginx base, que no tiene fallback SPA.
# Aporta try_files -> index.html para que las rutas client-side de react-router
# no devuelvan 404 al recargar/entrar directo. (DEUDA-NGINX-SPA)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Exponer el puerto 80 dentro del contenedor
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
