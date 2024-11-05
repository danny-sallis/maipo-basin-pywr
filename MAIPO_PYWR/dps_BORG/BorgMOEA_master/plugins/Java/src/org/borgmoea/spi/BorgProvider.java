package org.borgmoea.spi;

import org.moeaframework.core.Algorithm;
import org.moeaframework.core.Problem;
import org.moeaframework.core.spi.AlgorithmProvider;
import org.moeaframework.util.TypedProperties;

/**
 * Enables the Borg MOEA to be instantiated within the MOEA Framework.
 */
public class BorgProvider extends AlgorithmProvider {

	@Override
	public Algorithm getAlgorithm(String name, TypedProperties properties, Problem problem) {
		if (name.equalsIgnoreCase("borg")) {
			return new BorgAlgorithm(problem, properties);
		}
		
		return null;
	}

}
