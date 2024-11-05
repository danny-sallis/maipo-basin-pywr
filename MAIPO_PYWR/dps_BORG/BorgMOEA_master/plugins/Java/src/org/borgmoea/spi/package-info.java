/**
 * Enables the Borg MOEA to be instantiated within the MOEA Framework
 * by defining an {@code AlgorithmProvider} for the Borg MOEA.  The
 * classes inside this package should not be used directly, but rather
 * indirectly within the MOEA Framework:
 * 
 * <pre>
 * NondominatedPopulation result = new Executor()
 * 				.withProblem("UF1")
 * 				.withAlgorithm("Borg")
 * 				.withMaxEvaluations(10000)
 * 				.run();
 * </pre>
 * 
 * To enable this functionality, simply include the Borg MOEA JAR file
 * in the classpath alongside the MOEA Framework JARs.
 * 
 * Copyright 2014 David Hadka
 */
package org.borgmoea.spi;
