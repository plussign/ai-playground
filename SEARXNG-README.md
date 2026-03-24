# SearXNG Deployment (Docker)

## Files
- `docker-compose.searxng.yml`: Starts SearXNG and Redis (Valkey)
- `searxng/settings.yml`: SearXNG runtime configuration

## Start
Run in `e:\qwenTest`:

```powershell
docker compose -f docker-compose.searxng.yml up -d
```

Then open:
- `http://localhost:8080`

## Stop
```powershell
docker compose -f docker-compose.searxng.yml down
```

## Logs
```powershell
docker compose -f docker-compose.searxng.yml logs -f searxng
```

## Update SearXNG image
```powershell
docker compose -f docker-compose.searxng.yml pull
docker compose -f docker-compose.searxng.yml up -d
```

## Notes
- `server.secret_key` has been generated and written in `searxng/settings.yml`.
- If `docker` command is missing, install Docker Desktop first, then rerun the start command.
