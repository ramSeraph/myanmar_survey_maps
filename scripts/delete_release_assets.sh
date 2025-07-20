#!/bin/bash

# Check if a tag is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <tag>"
  exit 1
fi

TAG=$1

# Get the list of assets for the release
ASSETS=$(gh release view "$TAG" --json assets -q '.assets[].name')

if [ -z "$ASSETS" ]; then
  echo "No assets found for release $TAG."
  exit 0
fi

# Confirmation prompt
echo "The following assets will be deleted from release $TAG:"
echo "$ASSETS"
read -p "Are you sure you want to delete all these assets? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

# Loop through and delete each asset
echo "$ASSETS" | while read -r asset; do
  if [ -n "$asset" ]; then
    echo "Deleting asset: $asset"
    gh release delete-asset "$TAG" "$asset" -y
  fi
done

echo "All assets deleted from release $TAG."
