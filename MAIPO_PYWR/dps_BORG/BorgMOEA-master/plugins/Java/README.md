# Java Plugin

The Borg MOEA plugin for Java integrates into the [MOEA Framework](https://github.com/MOEAFramework/MOEAFramework).
The following steps are required to setup this plugin.

1. Download the latest [MOEA Framework Release](https://github.com/MOEAFramework/MOEAFramework/releases)
2. Extract the contents of `MOEAFramework-<version>.tar.gz` and copy all JAR files from the `lib/` directory into the `lib/` directory in this folder
3. Run `./build_native.sh` to compile the Borg MOEA shared library for the current operating system.
4. Run `ant run` to run the DTLZ2 example

## Distributing

Use `ant jar` to bundle all the code into a JAR that can be easily shared.  Please note that the JAR will only work on the
platform used when compiling the shared library.

To create a cross-platform JAR, run `./build_native.sh` on each operating system and architecture you wish to support.  The
library for each platform are generated inside the `native/` folder, such as `native/linux-x86-64/libborg.so`.  Combine all
of these into a single `native/` folder before running `ant jar`.
