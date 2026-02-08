FROM redis:7-alpine
EXPOSE 6379
CMD ["redis-server", "--save", "60", "1", "--loglevel", "warning", "--maxmemory", "50mb", "--maxmemory-policy", "allkeys-lru"]
