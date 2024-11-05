package org.borgmoea;

/**
 * A solution to the optimization problem.
 */
public class Solution {
	
	/**
	 * A reference to the underlying C solution.
	 */
	private BorgLibrary.BORG_Solution solution;
	
	/**
	 * Constructs a solution given a reference to the underlying C solution.
	 * 
	 * @param solution a reference to the underlying C solution
	 */
	Solution(BorgLibrary.BORG_Solution solution) {
		super();
		this.solution = solution;
	}
	
	/**
	 * Returns the decision variable at the given index.
	 * 
	 * @param index the index
	 * @return the decision variable at the given index
	 * @throws IndexOutOfBoundsException if the index if not valid
	 */
	public double getVariable(int index) {
		if ((index < 0) || (index > getNumberOfVariables())) {
			throw new IndexOutOfBoundsException();
		} else {
			return BorgLibrary.BORG_Solution_get_variable(solution, index);
		}
	}
	
	/**
	 * Returns the objective value at the given index.
	 * 
	 * @param index the index
	 * @return the objective value at the given index
	 * @throws IndexOutOfBoundsException if the index is not valid
	 */
	public double getObjective(int index) {
		if ((index < 0) || (index > getNumberOfObjectives())) {
			throw new IndexOutOfBoundsException();
		} else {
			return BorgLibrary.BORG_Solution_get_objective(solution, index);
		}
	}
	
	/**
	 * Returns the constraint value at the given index.
	 * 
	 * @param index the index
	 * @return the constraint value at the given index
	 * @throws IndexOutOfBoundsException if the index is not valid
	 */
	public double getConstraint(int index) {
		if ((index < 0) || (index > getNumberOfConstraints())) {
			throw new IndexOutOfBoundsException();
		} else {
			return BorgLibrary.BORG_Solution_get_constraint(solution, index);
		}
	}
	
	/**
	 * Returns the number of decision variables in the optimization problem.
	 * 
	 * @return the number of decision variables in the optimization problem
	 */
	public int getNumberOfVariables() {
		BorgLibrary.BORG_Problem problem = BorgLibrary.BORG_Solution_get_problem(solution);
		return BorgLibrary.BORG_Problem_number_of_variables(problem);
	}
	
	/**
	 * Returns the number of objectives in the optimization problem.
	 * 
	 * @return the number of objectives in the optimization problem
	 */
	public int getNumberOfObjectives() {
		BorgLibrary.BORG_Problem problem = BorgLibrary.BORG_Solution_get_problem(solution);
		return BorgLibrary.BORG_Problem_number_of_objectives(problem);
	}
	
	/**
	 * Returns the number of constraints in the optimization problem.
	 * 
	 * @return the number of constraints in the optimization problem
	 */
	public int getNumberOfConstraints() {
		BorgLibrary.BORG_Problem problem = BorgLibrary.BORG_Solution_get_problem(solution);
		return BorgLibrary.BORG_Problem_number_of_constraints(problem);
	}
	
	/**
	 * Returns {@code true} if this solution violates one or more constraints; {@code false} otherwise.
	 * 
	 * @return {@code true} if this solution violates one or more constraints; {@code false} otherwise
	 */
	public boolean violatesConstraints() {
		return BorgLibrary.BORG_Solution_violates_constraints(solution) != 0;
	}

}
