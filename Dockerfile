version: '3.8'

services:
  postgres-db:
    image: pgvector/pgvector:pg16
    container_name: postgres-db
    environment:
      POSTGRES_PASSWORD: pg123456
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
volumes:
  postgres-data:
    name: pg-data