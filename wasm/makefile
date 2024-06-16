build:
	emcc AutoJs.cpp -o AutoJsEngine.js -sMODULARIZE -sEXPORTED_RUNTIME_METHODS=ccall

run:

	emcc AutoJs.cpp -o AutoJsEngine.js -sMODULARIZE -sEXPORTED_RUNTIME_METHODS=ccall -s EXPORTED_FUNCTIONS="['_malloc', '_free']"
	node AutoJs.js
