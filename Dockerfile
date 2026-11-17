# TVCache server. Built only when INSTALL_TVCACHE=true; otherwise this stage
# writes an empty placeholder so the copy below still succeeds.
ARG INSTALL_TVCACHE=false
FROM rust:1.85-slim AS tvcache-rust
ARG INSTALL_TVCACHE=false
RUN mkdir -p /opt/tvcache/bin && \
    if [ "$INSTALL_TVCACHE" = "true" ]; then \
      apt-get update && apt-get install -y git pkg-config libssl-dev && \
      rm -rf /var/lib/apt/lists/* && \
      git clone https://github.com/abhivijay96/tvcache-rust.git /tmp/tvcache-rust && \
      git -C /tmp/tvcache-rust checkout a5842617105d9061ff5c070c26ceedffb4a8bb26 && \
      cargo build --release --manifest-path /tmp/tvcache-rust/Cargo.toml && \
      cp /tmp/tvcache-rust/target/release/tvcache /opt/tvcache/bin/tvcache && \
      rm -rf /tmp/tvcache-rust; \
    else \
      : > /opt/tvcache/bin/tvcache; \
    fi

# Base image with Python 3.12
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3 and pip
RUN apt-get update && \
    apt-get install -y build-essential cmake curl git ninja-build python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install uv
RUN pip install uv

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
# If you encounter a checkpoint issue, try using following old version of pathways-utils.
# RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git@b72729bb152b7b3426299405950b3af300d765a9#egg=pathwaysutils
RUN pip install gcsfs
RUN pip install wandb

# Set the working directory
WORKDIR /app

# Copy scripts and requirements first to leverage Docker cache
COPY scripts/install_tunix_vllm_requirement.sh scripts/
COPY requirements/ requirements/

RUN bash scripts/install_tunix_vllm_requirement.sh

# Copy pyproject.toml, README.md, and CMakeLists.txt to install dependencies first
COPY pyproject.toml README.md CMakeLists.txt /app/
RUN mkdir /app/tunix && touch /app/tunix/__init__.py
RUN uv pip install ".[tpu]"

# Install SFT/MaxText dependencies (unconditional)
RUN uv pip install --upgrade flax && \
    uv pip install torchax aqtp tokamax math_verify drjax && \
    uv pip install --no-deps git+https://github.com/google/maxtext.git

# Build argument to conditionally install Kubernetes tools
ARG INSTALL_K8S_TOOLS=false

# Install gcloud, kubectl, k9s
RUN if [ "$INSTALL_K8S_TOOLS" = "true" ]; then \
      apt-get update && \
      apt-get install -y vim lsof procps apt-transport-https ca-certificates gnupg && \
      (echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list) && \
      (curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --batch --yes --no-tty --dearmor -o /usr/share/keyrings/cloud.google.gpg) && \
      apt-get update && apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin kubectl && \
      (curl -sS https://webinstall.dev/k9s | bash) && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# Build argument to conditionally install DeepSWE evaluation dependencies
ARG INSTALL_DEEPSWE_DEPS=false

# Install DeepSWE specific dependencies and apply runtime patches conditionally
RUN if [ "$INSTALL_DEEPSWE_DEPS" = "true" ]; then \
      uv pip install kubernetes gym swebench==3.0.2 && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=clients/python/agentic-sandbox-client && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl && \
      uv pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
      sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
      sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py; \
    fi

# TVCache client and server binary. Off by default. The binary is copied from
# the rust stage; only this flag installs it and the pinned tvclient.
ARG INSTALL_TVCACHE=false
COPY --from=tvcache-rust /opt/tvcache/bin/tvcache /opt/tvcache/bin/tvcache
RUN if [ "$INSTALL_TVCACHE" = "true" ]; then \
      chmod 755 /opt/tvcache/bin/tvcache && \
      uv pip install "tvclient @ git+https://github.com/abhivijay96/tvclient.git@45d8c7d76dffca8d01a47315d14a82c4af9405f0"; \
    else \
      rm -rf /opt/tvcache; \
    fi

# Build argument to conditionally install MaxText dependencies
ARG INSTALL_MAXTEXT=false

# Install MaxText specific dependencies conditionally
RUN if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      uv pip install -r /app/requirements/maxtext_requirements.txt --torch-backend=cpu; \
fi

# Build argument to conditionally install Raiden weight sync dependencies
ARG INSTALL_RAIDEN=false
ARG RAIDEN_WHEEL_DIR=/app/raiden_wheels

# Install Raiden specific dependencies conditionally. Off by default, so a
# plain build never needs access to the wheel. The installer prefers a locally
# built wheel from ${RAIDEN_WHEEL_DIR} (see scripts/build_raiden_wheel.sh) and
# otherwise fetches the wheel pinned in scripts/install_raiden.sh.
COPY scripts/install_raiden.sh /app/scripts/
COPY raiden_wheels/ ${RAIDEN_WHEEL_DIR}/
RUN if [ "$INSTALL_RAIDEN" = "true" ]; then \
      RAIDEN_WHEEL_DIR="$RAIDEN_WHEEL_DIR" bash /app/scripts/install_raiden.sh; \
    fi

# Force install numpy version to avoid version conflicts.
RUN uv pip install numpy==2.3.5

# Copy the rest of the project files
COPY . .

# Compile proto buffer for discovery service, this has to be the last step.
RUN uv pip install grpcio-tools
RUN cd /app && find tunix/dist_rl/distributed -name "*.proto" -exec python -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app {} +

# Install Tunix in editable mode
RUN uv pip install --no-deps -e .



# Set the default command to bash
CMD ["bash"]
