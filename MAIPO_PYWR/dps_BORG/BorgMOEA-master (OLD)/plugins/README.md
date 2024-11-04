# Borg MOEA Plugins

These plugins allow integrating the Borg MOEA in various programming languages.  All of these plugins use
the shared library (`libborg.so`) on Linux or dynamic-link library (`borg.dll`) on Windows.

## Supported Languages

* [C#](C%23)
* [Java](Java)
* [Matlab](Matlab)
* [Python](Python)
* [R](R)

## Compiling the Shared Library

Practically any C/C++ compiler can be used to compile the Borg MOEA.  Here are some options:

### GNU C Compiler (gcc)

```bash

# Windows
gcc -shared -fPIC -O3 -o borg.dll borg.c mt19937ar.c -lm

# Linux
gcc -shared -fPIC -O3 -o libborg.so borg.c mt19937ar.c -lm
```

### Microsoft Visual Studio (GUI)

From the Visual Studio UI:

1. Create a new, empty project and add the files `borg.c`, `borg.h`, `mt19937ar.c`, and `mt19937ar.h`
2. Edit the project properties
3. On the General page:
   1. Set **Target Name** to `Borg`
   2. Set **Target Extension** to `dll`
   3. Set **Configuration Type** to `Dynamic Library (.dll)`
3. On the C/C++ Preprocessor page, add `BORG_EXPORTS` and `_CRT_SECURE_NO_WARNINGS` to the **Preprocessor Definitions**
4. On the C/C++ Advanced page, set **Compile As** to `Compile as C++ Code (/TP)`
5. Build the project

### Microsoft Visual C++ Compiler (MSVC)

Alternatively, we can invoke the compiler from the command prompt.  If you have Visual Studio installed and the `cl.exe`
program is not found, try using the **x64 Native Tools Command Prompt for VS 20XX**, found in the Start menu.

```bash

cl /LD /TP /DBORG_EXPORTS borg.c mt19937ar.c
```

