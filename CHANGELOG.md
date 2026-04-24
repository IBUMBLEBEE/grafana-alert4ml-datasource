# Changelog
# 0.2.8 (2026-04-24)

### Features and enhancements
- Merge pull request #12 from IBUMBLEBEE/ci

ci pipeline
- Ci(changelog): align cliff.toml format with Grafana CHANGELOG style

Update cliff.toml body template:
- Version header: `# X.Y.Z (YYYY-MM-DD)` (single #, parenthesised date)
- Section names: `### Features and enhancements` / `### Bug fixes` /
  `### Breaking changes` (plain English, no emoji)
- Commit entry: `- **Scope:** description` when scope is present
- Breaking changes detected by trailing ! or BREAKING CHANGE in body
- Ci: add commit-msg hook for conventional commits validation

Add .githooks/commit-msg shell script to enforce Conventional Commits
format on every git commit. Allowed types: feat | fix | perf | refactor |
docs | style | test | ci | chore.

Add prepare script in package.json so that npm install automatically
activates the hook via `git config core.hooksPath .githooks`.
- Ci(release): add git-cliff powered release workflow

- Rewrite scripts/release-pre.sh to use `npm version` for syncing
  package.json and package-lock.json, then call git-cliff to regenerate
  the full CHANGELOG.md before tagging

- Add cliff.toml with emoji-labelled commit groups (Features, Bug Fixes,
  Performance, Refactoring, Documentation, CI/CD, etc.), breaking-change
  annotation, and a Full Changelog compare link appended to each release

- Update .github/workflows/release.yml:
  - Add validate job that gates the build on tag == package.json version
  - Make build-frontend depend on validate
  - In release job: full git clone (fetch-depth: 0) + install git-cliff
    via taiki-e/install-action, generate per-release notes with
    `git cliff --latest --strip all` and feed them to softprops/action-gh-release
    so GitHub Release page shows structured changelog content
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.2.7...v0.2.8
# 0.2.7 (2026-04-23)

### Features and enhancements
- Chore: bump version to 0.2.7
- Improve UI display issues in the History TimeRange component
- Bug fixes: forecast training data leakage
- Add evaluation metrics
- Add test dataset
- Merge pull request #11 from IBUMBLEBEE/feat/skills

add skills
- Add skills
- Merge pull request #10 from IBUMBLEBEE/feat/ffi_interface

Optimize FFI interface
- Optimize FFI interface
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.2.6...v0.2.7
# 0.2.6 (2026-04-18)

### Bug fixes
- Fix WARNING

### Features and enhancements
- Chore: bump version to 0.2.6
- Merge pull request #9 from IBUMBLEBEE/feat/pipeline-optimization

update dynamics baseline algo
- Optimized some UI layouts
- Update dynamics baseline algo
- Update README.md
- Update README.md
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.2.5...v0.2.6
# 0.2.5 (2026-04-06)

### Features and enhancements
- Add arm64 arch
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.2.4...v0.2.5
# 0.2.4 (2026-04-06)

### Bug fixes
- Fix ci
- Fix ci: only amd64 build
- Fix ci error for ziglang download

### Features and enhancements
- Chore: bump version to 0.2.4
- Chore: bump version to 2.0.3
- Chore: bump version to 2.0.1
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v2.0.0...v0.2.4
# 2.0.0 (2026-04-06)

### Features and enhancements
- Chore: bump version to 2.0.0
- Merge pull request #8 from IBUMBLEBEE/feat/no-cgo

Static compilation
- Use zig cc and cargo zigbuild
- Static compilation using musl
- Static compilation
- Update README.md
- Update image files
- Merge pull request #6 from IBUMBLEBEE/feat/classifier

Add Time Series Classifier
- Add some algorithms
- Merge branch 'feat/classifier' of https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource into feat/classifier
- Add Time Series Classifier
- Add Time Series Classifier
- Merge pull request #5 from IBUMBLEBEE/feat/algo-select

Add selection features corresponding to the data and algorithms
- Add selection features corresponding to the data and algorithms
- Merge pull request #4 from IBUMBLEBEE/feat/external-storage

Feat/external storage
- UI Optimization and Adjustments
- Render the selected datasource UI using dynamic nesting techniques
- Optimize the code structure of the FFI section
- Remove README.dm
- Remove README.dm
- Move the common and core parts of multiple projects to the core project in advance
- Add Postgresql
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.24...v2.0.0
# 0.1.24 (2026-03-07)

### Features and enhancements
- Chore: bump version to 0.1.24
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.23...v0.1.24
# 0.1.23 (2026-03-06)

### Features and enhancements
- Update release.yml
- Update README.md
- Remove src/README.md
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.22...v0.1.23
# 0.1.22 (2026-03-05)

### Bug fixes
- Fix bug

### Features and enhancements
- Delete build.bat
- Update release.yml
- Update README.md
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.21...v0.1.22
# 0.1.21 (2026-03-05)

### Bug fixes
- Fix ci
- Fix ci
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.20...v0.1.21
# 0.1.20 (2026-03-05)

### Bug fixes
- Fix ci
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.19...v0.1.20
# 0.1.19 (2026-03-04)

### Features and enhancements
- Merge pull request #3 from IBUMBLEBEE/ffi

ffi
- Arrow ffi
- Ffi init
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.18...v0.1.19
# 0.1.18 (2026-02-27)

### Features and enhancements
- Update README.md
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.17...v0.1.18
# 0.1.17 (2026-01-29)

### Bug fixes
- Fix: github ci testing
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.16...v0.1.17
# 0.1.16 (2026-01-29)

### Bug fixes
- Fix: build plugin for go build
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.15...v0.1.16
# 0.1.15 (2026-01-29)

### Bug fixes
- Fix: go test failed
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.14...v0.1.15
# 0.1.14 (2026-01-29)

### Bug fixes
- Fix: github ci - go version 1.25
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.13...v0.1.14
# 0.1.13 (2026-01-29)

### Bug fixes
- Fix: go version use 1.21
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.12...v0.1.13
# 0.1.12 (2026-01-29)

### Bug fixes
- Fix: github ci for grafana build package
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.11...v0.1.12
# 0.1.11 (2026-01-29)

### Bug fixes
- Fix: using PAT token
- Fix: github ci failed - grpc proto deps
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.9...v0.1.11
# 0.1.9 (2026-01-29)

### Bug fixes
- Fix: github ci testing - rename rust builder name
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.8...v0.1.9
# 0.1.8 (2026-01-29)

### Bug fixes
- Fix: remove grafana builder
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.7...v0.1.8
# 0.1.7 (2026-01-29)

### Bug fixes
- Fix: github ci testing - update Makefile
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.6...v0.1.7
# 0.1.6 (2026-01-29)

### Bug fixes
- Fix: github ci testing for rust builder
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.5...v0.1.6
# 0.1.5 (2026-01-29)

### Bug fixes
- Fix: github ci testing
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.4...v0.1.5
# 0.1.4 (2026-01-25)

### Features and enhancements
- Bug fixes: not permitted to access the file system
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.3...v0.1.4
# 0.1.3 (2026-01-25)

### Features and enhancements
- V0.1.3 release
- Upgrade package.json for CVE
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.2...v0.1.3
# 0.1.2 (2026-01-25)

### Features and enhancements
- Release v0.1.2
- Merge pull request #2 from IBUMBLEBEE/multi-process-arch

Multi process arch
- Update Makefile
- Add baseline and outlier for grpc server
- V0.1.0 for multi process arch
- Add rust grpc server
- Add grpc sever/client
- Add grpc client
- Add grpc server
- Patch for  musl build: missing qsort_r symbol
- Supports static compilation with musl
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.1...v0.1.2
# 0.1.1 (2026-01-21)

### Features and enhancements
- Update CHANGELOG.md
- V0.1.1
- Update README.md
- Update README.md
- Using SQLite in memory mode
- Move rsod
- Upgrade grafana-plugin-sdk-go version to  v0.285.0
- Update README.md
**Full Changelog**: https://github.com/IBUMBLEBEE/grafana-alert4ml-datasource/compare/v0.1.0...v0.1.1
# 0.1.0 (2026-01-20)

### Features and enhancements
- Update README.md
- First commit
- Create README.md

