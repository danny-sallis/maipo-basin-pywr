# Matlab Plugin

The Borg MOEA plugin for Matlab will use the `mex` compiler to produce a "mex file".

## Windows (with Matlab)

First, follow the [Compiling the Shared Library](..#compiling-the-shared-library) instructions using
**Visual Studio**.  Copy `Borg.dll` and `Borg.lib` into this folder.  Then, run:

```bash

mex nativeborg.cpp Borg.lib
```

This will create the file `nativeborg.mexw64`.  All of these files must be in this folder.
Finally, start Matlab and run the command:

```matlab

[vars, objs] = borg(11, 2, 0, @DTLZ2, 10000, [0.01, 0.01])
```

## Linux (with Octave)

On Linux, we will use Octave, an open-source alternative to MATLAB.  First, we compile
the shared library:

```bash

gcc -shared -fPIC -O3 -o libborg.so borg.c mt19937ar.c -lm
```

Copy both `libborg.so` and `borg.h` into this folder.  Then, we can compile and run the example:

```bash

mkoctfile -mex -DOCTAVE -L. -lborg -Wl,-rpath,\. nativeborg.cpp
octave --eval "[vars, objs] = borg(11, 2, 0, @DTLZ2, 10000, [0.01, 0.01])"
```
