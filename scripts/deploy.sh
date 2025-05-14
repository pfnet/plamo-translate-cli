rm -rf dist
ARCHFLAGS="-arch arm64" MACOSX_DEPLOYMENT_TARGET="11.0" \
uv build --wheel

WHEEL_FILENAME=$(ls dist/plamo_translate-*.whl)
uv run -m wheel tags \
    --python-tag py3 \
    --abi-tag none \
    --platform-tag macosx_11_0_arm64 \
    ${WHEEL_FILENAME}
rm -rf ${WHEEL_FILENAME}

uv run twine upload dist/*