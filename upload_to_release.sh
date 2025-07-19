#!/bin/bash

# A script to upload files from a folder to a GitHub release,
# skipping files that already exist in the release.

set -e
set -o pipefail

# --- Configuration ---
# TAG: The git tag of the release to upload to.
# FOLDER: The local folder containing the files to upload.

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <tag> <folder> <extension_without_the_leading_dot>"
    exit 1
fi

TAG="$1"
FOLDER="$2"
EXT="$3"

# --- Pre-flight Checks ---

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: gh command-line tool is not installed. Please install it to continue."
    exit 1
fi

if [ -z "$TAG" ]; then
    echo "Error: Release tag is not specified."
    exit 1
fi

if [ ! -d "$FOLDER" ]; then
    echo "Error: Folder '$FOLDER' not found."
    exit 1
fi

if [ -z "$EXT" ]; then
    echo "Error: Extension '$EXT' is not specified."
    exit 1
fi

# --- Main Logic ---

echo "Fetching existing assets for releases matching pattern '${TAG}(-extra[0-9]+)?'..."

# Get all releases, then filter for the base release and any -extra releases, and sort them
RELEASES_TO_PROCESS=$(gh release list --json tagName -q '.[].tagName' | grep -E "^${TAG}(-extra[0-9]+)?$" | sort -V)

if [ -z "$RELEASES_TO_PROCESS" ]; then
    echo "Error: No releases found matching pattern '${TAG}(-extra[0-9]+)?'." >&2
    exit 1
fi

echo "Found releases: $RELEASES_TO_PROCESS"

EXISTING_ASSETS=""
for release in $RELEASES_TO_PROCESS; do
  echo "Fetching assets from release: $release"
  assets=$(gh release view "$release" --json assets -q '.assets[].name')
  if [ -n "$assets" ]; then
    EXISTING_ASSETS="${EXISTING_ASSETS}"$'
'"${assets}"
  fi
done

if [ -z "$EXISTING_ASSETS" ]; then
    echo "Could not fetch any assets. This could mean the releases do not have assets, or you lack permissions."
fi

# Determine available releases and their current asset counts
AVAILABLE_RELEASES=()
AVAILABLE_ASSET_COUNTS=()
for release in $RELEASES_TO_PROCESS; do
    count=$(gh release view "$release" --json assets -q '.assets | length' 2>/dev/null || echo "0")
    
    MAX_ASSETS=998
    if [ "$release" == "$TAG" ]; then
        MAX_ASSETS=988
    fi

    if [ "$count" -lt "$MAX_ASSETS" ]; then
        AVAILABLE_RELEASES+=("$release")
        AVAILABLE_ASSET_COUNTS+=("$count")
    fi
done

echo "Starting upload process from folder '$FOLDER'..."

# Find all files in the folder that are not yet in any release and loop through them.
find "${FOLDER}" -type f | grep "^.*\.${EXT}$" | while read -r FILE_PATH; do
    FILENAME=$(basename "$FILE_PATH")

    if echo "$EXISTING_ASSETS" | grep -q "^${FILENAME}$"; then
        echo "  -> Skipping '$FILENAME', it already exists in a release."
        continue
    fi

    # Find a release to upload to
    if [ ${#AVAILABLE_RELEASES[@]} -eq 0 ]; then
        echo "Error: All existing releases are full. No space to upload '$FILENAME'." >&2
        exit 1
    fi

    UPLOAD_TARGET=${AVAILABLE_RELEASES[0]}

    echo "  -> Uploading '$FILENAME' to '$UPLOAD_TARGET'..."
    gh release upload "$UPLOAD_TARGET" "$FILE_PATH" --clobber=false

    # Update asset count and remove release from available list if full
    AVAILABLE_ASSET_COUNTS[0]=$((${AVAILABLE_ASSET_COUNTS[0]} + 1))
    
    MAX_ASSETS=998
    if [ "$UPLOAD_TARGET" == "$TAG" ]; then
        MAX_ASSETS=988
    fi

    if [ "${AVAILABLE_ASSET_COUNTS[0]}" -ge "$MAX_ASSETS" ]; then
        AVAILABLE_RELEASES=("${AVAILABLE_RELEASES[@]:1}")
        AVAILABLE_ASSET_COUNTS=("${AVAILABLE_ASSET_COUNTS[@]:1}")
    fi
done

echo "Upload process complete."
