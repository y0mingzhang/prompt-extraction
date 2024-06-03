#!/bin/bash

set -Eeuo pipefail

isort .
black .
ruff .
