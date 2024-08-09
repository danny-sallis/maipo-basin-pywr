# C# Plugin

Follow the [Compiling the Shared Library](..#compiling-the-shared-library) instructions to compile the
Borg MOEA shared library.  Then, copy the library into this folder before continuing below.

## Compiling and Running the Example

### Visual Studio (GUI)

First, we will create a project to build the dynamically-linked library:

1. Create a new project using the Visual C# template.  Choose **Class Library**.
2. Delete any default `.cs` files in the project.
3. Right-click the project, choose `Add -> Existing File`, and select `BorgMOEA.cs`
4. Build the project to produce `BorgMOEA.dll`

Then, we will create another project to run the example:

1. Create a new project using the Visual C# template.  Choose **Console Application**.
2. Delete any default `.cs` files in the project.
3. Right-click the project, choose `Add -> Existing File`, and select `Example.cs`
4. Right-click References, choose **Add Reference**, and select `BorgMOEA.dll`
5. Build the project to produce the executable
6. Copy `borg.dll` into the same folder as the executable
7. Run the executable!

### Visual Studio C# Compiler (csc.exe)

From the **x64 Native Tools Command Prompt for VS 20XX**, run:

```bash

csc -target:library BorgMOEA.cs
csc -reference:BorgMOEA.dll Example.cs
Example.exe
```

### Mono

Mono is an open-source version of the .NET Framework.  We can compile and run the DTLZ2
example with:

```bash

mcs -target:library BorgMOEA.cs
mcs -reference:BorgMOEA.dll Example.cs
mono Example.exe
```

## Troubleshooting

A **BadImageFormatException** occurs when the generated DLLs do not match the expected format,
typically due to the DLLs being compiled for a different architecture or bitness (32 vs 64 bit).

```
Unhandled Exception: System.TypeInitializationException: The type initializer for 'BorgMOEA.Borg' threw an exception.
---> System.BadImageFormatException: An attempt was made to load a program with an incorrect format. (Exception from HRESULT: 0x8007000B)
```

Make sure you are using the correct version of the compiler for your system.  For example, if using
the 64-bit version of C#, use the 64-bit version of the C/C++ compiler.  On Windows, this would mean
compiling inside the **x64 Native Tools Command Prompt for Visual Studio 20XX**.
