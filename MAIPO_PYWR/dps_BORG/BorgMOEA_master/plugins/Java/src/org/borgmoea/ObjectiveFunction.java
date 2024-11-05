package org.borgmoea;

/**
 * Interface for defining the objective and constraint calculations for an
 * optimization problem.
 */
public interface ObjectiveFunction {
	
	/**
	 * Evaluates the given decision variables to compute the objectives and constraints
	 * for an optimization problem.  The {@code variables} array is populated with the
	 * decision variable values when this method is invoked.  This method must assign
	 * the values in the {@code objectives} and {@code constraints} arrays before this
	 * method returns.  The {@code constraints} array will have length {@code 0} if no
	 * constraints are defined.
	 * <p>
	 * For each constraint, if the constraint is feasible, set the constraint value to
	 * {@code 0}.  All non-zero values indicate constraint violations.  In general,
	 * larger constraint values (both positive and negative) indicate larger or more
	 * severe constraint violations.
	 * 
	 * @param variables the array containing the decision variable values
	 * @param objectives the array for saving the objective values
	 * @param constraints the array for saving the constraint values
	 */
	public void evaluate(double[] variables, double[] objectives, double[] constraints);

}
