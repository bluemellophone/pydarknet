#################################
echo 'Removing old build'
#rm -rf build
#rm -rf CMakeFiles
#rm -rf CMakeCache.txt
#rm -rf cmake_install.cmake

python2.7 -c "import utool as ut; print('keeping build dir' if ut.get_argflag('--no-rmbuild') else ut.delete('build'))" $@
#################################
echo 'Creating new build'
mkdir -p build
cd build
#################################

if [[ "$(which nvcc)" == "" ]]; then
    export CMAKE_CUDA=Off
else
    export CMAKE_CUDA=On
fi

export PYEXE=$(which python2.7)
if [[ "$VIRTUAL_ENV" == ""  ]]; then
    export LOCAL_PREFIX=/usr/local
    export _SUDO="sudo"
else
    export LOCAL_PREFIX=$($PYEXE -c "import sys; print(sys.prefix)")/local
    export _SUDO=""
fi

echo 'Configuring with cmake'
if [[ '$OSTYPE' == 'darwin'* ]]; then
    export CONFIG="-DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_C_COMPILER=clang2 -DCMAKE_CXX_COMPILER=clang2++ -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV"
else
    export CONFIG="-DCMAKE_BUILD_TYPE='Release' -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV"
fi
export CONFIG="$CONFIG -DCUDA=$CMAKE_CUDA"
echo "$CONFIG"

cmake $CONFIG -G 'Unix Makefiles' ..
#################################
echo 'Building with make'
export NCPUS=$(grep -c ^processor /proc/cpuinfo)
make -j$NCPUS -w
#################################
echo 'Moving the shared library'
cp -v lib* ../pydarknet
cd ..
