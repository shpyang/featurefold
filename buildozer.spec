[app]
title = FeatureFold
package.name = featurefold
package.domain = org.t2igame
source.dir = .
source.include_exts = py,png,npz,json
source.exclude_dirs = desktop,tests,bin,.buildozer,__pycache__,.git
source.exclude_patterns = bake_assets.py,*.md
version = 0.1.0
requirements = python3,kivy==2.3.0,numpy
orientation = portrait
fullscreen = 0
android.api = 34
android.minapi = 24
android.archs = arm64-v8a
android.allow_backup = 0
icon.filename = icon.png
presplash.filename = presplash.png
android.release_artifact = aab
p4a.branch = v2024.01.21 
android.accept_sdk_license = True


[buildozer]
log_level = 2
warn_on_root = 1
