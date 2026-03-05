#!/bin/bash
# Script to update plugin version across all configuration files

if [ -z "$1" ]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 0.1.5"
  exit 1
fi

VERSION="$1"

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: Version must be in format X.Y.Z (e.g., 0.1.5)"
  exit 1
fi

echo "Updating plugin version to $VERSION..."

# Update package.json
if [ -f package.json ]; then
  sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${VERSION}\"/" package.json
  echo "✓ Updated package.json"
fi

echo "Version update complete! New version: $VERSION"
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Commit changes: git add . && git commit -m 'chore: bump version to $VERSION'"
echo "3. Create tag: git tag -a v$VERSION -m 'Release v$VERSION'"
echo "4. Push: git push origin main && git push origin v$VERSION"
