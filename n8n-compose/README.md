# n8n with SSL Setup

This setup uses Docker Compose to run n8n with HTTPS enabled.

## Prerequisites

Before running `docker-compose up`, you must create a `certs` directory in the project root (one level above this directory).

```sh
mkdir ../certs
```

This directory must contain your SSL certificate and private key:

-   `../certs/cert.pem`
-   `../certs/key.pem`

**Note:** The `certs` directory is intentionally not checked into version control for security reasons, as defined in the root `.gitignore` file. You must provide your own certificate and key.
