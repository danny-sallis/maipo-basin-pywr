package org.borgmoea;

import java.util.HashMap;
import java.util.Map;

import com.sun.jna.Native;
import com.sun.jna.NativeLong;
import com.sun.jna.Pointer;

/**
 * Solves an optimization problem using the Borg MOEA.
 */
public class Borg {
	
	/**
	 * A reference to the underlying C problem.
	 */
	private BorgLibrary.BORG_Problem problem;

	/**
	 * The callback function wrapper.
	 */
	private FunctionWrapper functionWrapper;
	
	/**
	 * The algorithm parameters set by the user.
	 */
	private Map<String, Number> settings;

	@SuppressWarnings("unused")
	private ObjectiveFunction function;
	
	/**
	 * Creates a new instance of the Borg MOEA.
	 * 
	 * @param numberOfVariables the number of decision variables in the optimization problem
	 * @param numberOfObjectives the number of objectives in the optimization problem
	 * @param numberOfConstraints the number of constraints in the optimization problem
	 * @param function the function defining the optimization problem
	 * @throws IllegalArgumentException if any of the arguments are invalid
	 */
	public Borg(int numberOfVariables, int numberOfObjectives, int numberOfConstraints, ObjectiveFunction function) {
		super();
		this.function = function;
		
		if (numberOfVariables < 1) {
			throw new IllegalArgumentException("Requires at least one decision variable");
		}
		
		if (numberOfObjectives < 1) {
			throw new IllegalArgumentException("Requires at least one objective");
		}
		
		if (numberOfConstraints < 0) {
			throw new IllegalArgumentException("Number of constraints can not be negative");
		}
		
		if (function == null) {
			throw new IllegalArgumentException("Function is null");
		}

		functionWrapper = new FunctionWrapper(function);
		problem = BorgLibrary.BORG_Problem_create(numberOfVariables, numberOfObjectives, numberOfConstraints, functionWrapper);
		settings = new HashMap<String, Number>();
	}
	
	@Override
	protected void finalize() throws Throwable {
		if (problem != null) {
			BorgLibrary.BORG_Problem_destroy(problem);
		}
		
		super.finalize();
	}

	/**
	 * Returns the number of decision variables in the optimization problem.
	 * 
	 * @return the number of decision variables in the optimization problem
	 */
	public int getNumberOfVariables() {
		return BorgLibrary.BORG_Problem_number_of_variables(problem);
	}
	
	/**
	 * Returns the number of objectives in the optimization problem.
	 * 
	 * @return the number of objectives in the optimization problem
	 */
	public int getNumberOfObjectives() {
		return BorgLibrary.BORG_Problem_number_of_objectives(problem);
	}
	
	/**
	 * Returns the number of constraints in the optimization problem.
	 * 
	 * @return the number of constraints in the optimization problem
	 */
	public int getNumberOfConstraints() {
		return BorgLibrary.BORG_Problem_number_of_constraints(problem);
	}
	
	/**
	 * Sets the decision variable lower and upper bounds.
	 * 
	 * @param lowerBounds the lower bounds
	 * @param upperBounds the upper bounds
	 * @throws IllegalArgumentException if the length of the lower or upper bounds arrays are incorrect
	 */
	public void setBounds(double[] lowerBounds, double[] upperBounds) {
		if (lowerBounds.length != getNumberOfVariables()) {
			throw new IllegalArgumentException("Incorrect number of bounds specified");
		}
		
		if (upperBounds.length != getNumberOfVariables()) {
			throw new IllegalArgumentException("Incorrect number of bounds specified");
		}
		
		for (int i = 0; i < lowerBounds.length; i++) {
			setBounds(i, lowerBounds[i], upperBounds[i]);
		}
	}
	
	/**
	 * Sets the lower and upper bounds for the given decision variable.
	 * 
	 * @param index the index of the decision variable
	 * @param lowerBound the lower bound
	 * @param upperBound the upper bound
	 * @throws IndexOutOfBoundsException if the index is not valid
	 */
	public void setBounds(int index, double lowerBound, double upperBound) {
		if ((index < 0) || (index >= getNumberOfVariables())) {
			throw new IndexOutOfBoundsException();
		} else {
			BorgLibrary.BORG_Problem_set_bounds(problem, index, lowerBound, upperBound);
		}
	}
	
	/**
	 * Sets the epsilons for the objective values.  The epsilons control the
	 * granularity / resolution of the Pareto optimal set.  Small epsilons
	 * typically result in larger Pareto optimal sets, but can reduce runtime
	 * performance.
	 * 
	 * @param epsilons the epsilon values
	 * @throws IllegalArgumentException if the length of the epsilon array is incorrect
	 */
	public void setEpsilons(double[] epsilons) {
		if (epsilons.length != getNumberOfObjectives()) {
			throw new IllegalArgumentException("Incorrect number of epsilons specified");
		}
		
		for (int i = 0; i < epsilons.length; i++) {
			setEpsilon(i, epsilons[i]);
		}
	}
	
	/**
	 * Sets the epsilon value for the given objective.
	 * 
	 * @param index the index of the objective
	 * @param epsilon the epsilon value
	 * @throws IndexOutOfBoundsException if the index is not valid
	 */
	public void setEpsilon(int index, double epsilon) {
		if ((index < 0) || (index >= getNumberOfObjectives())) {
			throw new IndexOutOfBoundsException();
		} else {
			BorgLibrary.BORG_Problem_set_epsilon(problem, index, epsilon);
		}
	}
	
	/**
	 * Returns the parameter value as a double.
	 * 
	 * @param name the parameter name
	 * @param defaultValue the default value if the parameter is not set
	 * @return the parameter value
	 */
	private double lookupDouble(String name, double defaultValue) {
		if (settings.containsKey(name)) {
			return settings.get(name).doubleValue();
		} else {
			return defaultValue;
		}
	}
	
	/**
	 * Returns the parameter value as an integer.
	 * 
	 * @param name the parameter name
	 * @param defaultValue the default value if the parameter is not set
	 * @return the parameter value
	 */
	private int lookupInt(String name, int defaultValue) {
		if (settings.containsKey(name)) {
			return settings.get(name).intValue();
		} else {
			return defaultValue;
		}
	}
	
	/**
	 * Sets the value of a parameter.  The name should match one
	 * of the parameters defined by the C Borg API.  Default parameter values
	 * are used for any undefined parameters.
	 * 
	 * @param name the parameter name
	 * @param value the parameter value
	 */
	public void setParameter(String name, double value) {
		settings.put(name, value);
	}
	
	/**
	 * Sets the value of a parameter.  The name should match one
	 * of the parameters defined by the C Borg API.  Default parameter values
	 * are used for any undefined parameters.
	 * 
	 * @param name the parameter name
	 * @param value the parameter value
	 */
	public void setParameter(String name, int value) {
		settings.put(name, value);
	}

	private BorgLibrary.BORG_Algorithm algorithm;
	private BorgLibrary.BORG_Operator pm;
	private BorgLibrary.BORG_Operator sbx;
	private BorgLibrary.BORG_Operator de;
	private BorgLibrary.BORG_Operator um;
	private BorgLibrary.BORG_Operator spx;
	private BorgLibrary.BORG_Operator pcx;
	private BorgLibrary.BORG_Operator undx;

	public void create() {
		pm = BorgLibrary.BORG_Operator_create("PM", 1, 1, 2, BorgLibrary.JNA_NATIVE_LIB.getFunction("BORG_Operator_PM"));
		BorgLibrary.BORG_Operator_set_parameter(pm, 0, lookupDouble("pm.rate", 1.0 / getNumberOfVariables()));
		BorgLibrary.BORG_Operator_set_parameter(pm, 1, lookupDouble("pm.distributionIndex", 20.0));
				
		sbx = BorgLibrary.BORG_Operator_create("SBX", 2, 2, 2, BorgLibrary.JNA_NATIVE_LIB.getFunction("BORG_Operator_SBX"));
		BorgLibrary.BORG_Operator_set_parameter(sbx, 0, lookupDouble("sbx.rate", 1.0));
		BorgLibrary.BORG_Operator_set_parameter(sbx, 1, lookupDouble("sbx.distributionIndex", 15.0));
		BorgLibrary.BORG_Operator_set_mutation(sbx, pm);

		de = BorgLibrary.BORG_Operator_create("DE", 4, 1, 2, BorgLibrary.JNA_NATIVE_LIB.getFunction("BORG_Operator_DE"));
		BorgLibrary.BORG_Operator_set_parameter(de, 0, lookupDouble("de.crossoverRate", 0.1));
		BorgLibrary.BORG_Operator_set_parameter(de, 1, lookupDouble("de.stepSize", 0.5));
		BorgLibrary.BORG_Operator_set_mutation(de, pm);

		um = BorgLibrary.BORG_Operator_create("UM", 1, 1, 1, BorgLibrary.JNA_NATIVE_LIB.getFunction("BORG_Operator_UM"));
		BorgLibrary.BORG_Operator_set_parameter(um, 0, lookupDouble("um.rate", 1.0 / getNumberOfVariables()));

		spx = BorgLibrary.BORG_Operator_create("SPX", lookupInt("spx.parents", 10), lookupInt("spx.offspring", 2), 1, BorgLibrary.JNA_NATIVE_LIB.getFunction("BORG_Operator_SPX"));
		BorgLibrary.BORG_Operator_set_parameter(spx, 0, lookupDouble("spx.epsilon", 3.0));

		pcx = BorgLibrary.BORG_Operator_create("PCX", lookupInt("pcx.parents", 10), lookupInt("pcx.offspring", 2), 2, BorgLibrary.JNA_NATIVE_LIB.getFunction("BORG_Operator_PCX"));
		BorgLibrary.BORG_Operator_set_parameter(pcx, 0, lookupDouble("pcx.eta", 0.1));
		BorgLibrary.BORG_Operator_set_parameter(pcx, 1, lookupDouble("pcx.zeta", 0.1));

		undx = BorgLibrary.BORG_Operator_create("UNDX", lookupInt("undx.parents", 10), lookupInt("undx.offspring", 2), 2, BorgLibrary.JNA_NATIVE_LIB.getFunction("BORG_Operator_UNDX"));
		BorgLibrary.BORG_Operator_set_parameter(undx, 0, lookupDouble("undx.zeta", 0.5));
		BorgLibrary.BORG_Operator_set_parameter(undx, 1, lookupDouble("undx.eta", 0.35));

		algorithm = BorgLibrary.BORG_Algorithm_create(problem, 6);
		BorgLibrary.BORG_Algorithm_set_operator(algorithm, 0, sbx);
		BorgLibrary.BORG_Algorithm_set_operator(algorithm, 1, de);
		BorgLibrary.BORG_Algorithm_set_operator(algorithm, 2, pcx);
		BorgLibrary.BORG_Algorithm_set_operator(algorithm, 3, spx);
		BorgLibrary.BORG_Algorithm_set_operator(algorithm, 4, undx);
		BorgLibrary.BORG_Algorithm_set_operator(algorithm, 5, um);

		BorgLibrary.BORG_Algorithm_set_initial_population_size(algorithm, lookupInt("initialPopulationSize", 100));
		BorgLibrary.BORG_Algorithm_set_minimum_population_size(algorithm, lookupInt("minimumPopulationSize", 100));
		BorgLibrary.BORG_Algorithm_set_maximum_population_size(algorithm, lookupInt("maximumPopulationSize", 10000));
		BorgLibrary.BORG_Algorithm_set_population_ratio(algorithm, 1.0 / lookupDouble("injectionRate", 0.25));
		BorgLibrary.BORG_Algorithm_set_selection_ratio(algorithm, lookupDouble("selectionRatio", 0.02));
		BorgLibrary.BORG_Algorithm_set_restart_mode(algorithm, lookupInt("restartMode", BorgLibrary.BORG_Restart.RESTART_DEFAULT));
		BorgLibrary.BORG_Algorithm_set_max_mutation_index(algorithm, lookupInt("maxMutationIndex", 10));
		BorgLibrary.BORG_Algorithm_set_probability_mode(algorithm, lookupInt("probabilityMode", BorgLibrary.BORG_Probabilities.PROBABILITIES_DEFAULT));
	}

	public Result getResult() {
		BorgLibrary.BORG_Archive result = BorgLibrary.BORG_Algorithm_get_result(algorithm);
		return new Result(result, null);
	}

	public void step() {
		BorgLibrary.BORG_Algorithm_step(algorithm);
	}
	
	public Result solve(int maxEvaluations) {
		create();
		
		while (getNumberOfEvaluations() < maxEvaluations) {
			BorgLibrary.BORG_Algorithm_step(algorithm);
		}
		
		return getResult();
	}

	public int getNumberOfEvaluations() {
		return BorgLibrary.BORG_Algorithm_get_nfe(algorithm);
	}

	public void destroy() {
		BorgLibrary.BORG_Operator_destroy(sbx);
		BorgLibrary.BORG_Operator_destroy(de);
		BorgLibrary.BORG_Operator_destroy(pm);
		BorgLibrary.BORG_Operator_destroy(um);
		BorgLibrary.BORG_Operator_destroy(spx);
		BorgLibrary.BORG_Operator_destroy(pcx);
		BorgLibrary.BORG_Operator_destroy(undx);
		BorgLibrary.BORG_Algorithm_destroy(algorithm);
	}
	
	/**
	 * Wraps an {@link ObjectiveFunction} so that the internal JNA API for
	 * defining function pointers is hidden from the user.
	 */
	private class FunctionWrapper implements BorgLibrary.BORG_Problem_create_function_callback {
		
		/**
		 * The function defining the optimization problem.
		 */
		private ObjectiveFunction function;
		
		/**
		 * Constructs a new objective function wrapper.
		 * 
		 * @param function the function defining the optimization problem
		 */
		public FunctionWrapper(ObjectiveFunction function) {
			super();
			this.function = function;
		}
		
		/**
		 * Copies the values from a C array to a Java array.
		 * 
		 * @param pointer the C array pointer
		 * @param array the Java array
		 */
		public void fromPointer(Pointer pointer, double[] array) {
			for (int i = 0; i < array.length; i++) {
				array[i] = pointer.getDouble(i * Native.getNativeSize(Double.TYPE));
			}
		}
		
		/**
		 * Copies the values from a Java array to a C array.
		 *  
		 * @param array the Java array
		 * @param pointer the C array pointer
		 */
		public void toPointer(double[] array, Pointer pointer) {
			for (int i = 0; i < array.length; i++) {
				pointer.setDouble(i * Native.getNativeSize(Double.TYPE), array[i]);
			}
		}
		
		@Override
		public void apply(Pointer doublePtr1, Pointer doublePtr2,
				Pointer doublePtr3) {
			double[] variables = new double[getNumberOfVariables()];
			double[] objectives = new double[getNumberOfObjectives()];
			double[] constraints = new double[getNumberOfConstraints()];
			
			fromPointer(doublePtr1, variables);
			
			function.evaluate(variables, objectives, constraints);
			
			toPointer(objectives, doublePtr2);
			toPointer(constraints, doublePtr3);
		}
		 
	}
	
	/**
	 * Sets the pseudo-random number generator seed.
	 * 
	 * @param value the seed value
	 */
	public static void setSeed(int value) {
		BorgLibrary.BORG_Random_seed(new NativeLong(value));
	}
	
	/**
	 * Enables debugging output from the Borg MOEA.
	 */
	public static void enableDebugging() {
		BorgLibrary.BORG_Debug_on();
	}
	
	/**
	 * Disables debugging output from the Borg MOEA.
	 */
	public static void disableDebugging() {
		BorgLibrary.BORG_Debug_off();
	}

}
