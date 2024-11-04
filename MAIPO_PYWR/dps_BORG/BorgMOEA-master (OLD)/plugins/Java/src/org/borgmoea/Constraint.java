package org.borgmoea;

/**
 * Helper functions for defining constraints.  These functions ensure several
 * conditions hold.  First, if the constraint is satisfied, the value is 0.
 * If the constraint is violated, then the value is non-zero and will scale
 * linearly with the degree of violation.
 */
public class Constraint {
	
	/**
	 * The minimum constraint value when a constraint is violated.  This
	 * ensures all constraint violations are distinct from 0.0 even if the
	 * actual constraint violation is approaching machine precision.
	 */
	public static double PRECISION = 0.1;
	
	/**
	 * Private constructor to prevent instantiation.
	 */
	private Constraint() {
		super();
	}
	
	/**
	 * Defines the constraint x > y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double greaterThan(double x, double y, double epsilon) {
		return x > y-epsilon ? 0.0 : y-x+PRECISION;
	}
	
	/**
	 * Defines the constraint x > y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @return the constraint value
	 */
	public static double greaterThan(double x, double y) {
		return greaterThan(x, y, 0.0);
	}
	
	/**
	 * Defines the constraint x < y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double lessThan(double x, double y, double epsilon) {
		return x < y+epsilon ? 0.0 : x-y+PRECISION;
	}
	
	/**
	 * Defines the constraint x < y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @return the constraint value
	 */
	public static double lessThan(double x, double y) {
		return lessThan(x, y, 0.0);
	}

	/**
	 * Defines the constraint x >= y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double greaterThanOrEqual(double x, double y, double epsilon) {
		return x >= y-epsilon ? 0.0 : y-x+PRECISION;
	}

	/**
	 * Defines the constraint x >= y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @return the constraint value
	 */
	public static double greaterThanOrEqual(double x, double y) {
		return greaterThanOrEqual(x, y, 0.0);
	}
	
	/**
	 * Defines the constraint x <= y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double lessThanOrEqual(double x, double y, double epsilon) {
		return x <= y+epsilon ? 0.0 : x-y+PRECISION;
	}

	/**
	 * Defines the constraint x <= y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @return the constraint value
	 */
	public static double lessThanOrEqual(double x, double y) {
		return lessThanOrEqual(x, y, 0.0);
	}
	
	/**
	 * Defines the constraint x == y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double equal(double x, double y, double epsilon) {
		return Math.abs(y-x) < epsilon ? 0.0 : Math.abs(y-x)+PRECISION;
	}

	/**
	 * Defines the constraint x == y.
	 * 
	 * @param x the first value
	 * @param y the second value
	 * @return the constraint value
	 */
	public static double equal(double x, double y) {
		return equal(x, y, 0.0);
	}
	
	/**
	 * Defines the constraint x == 0.
	 * 
	 * @param x the first value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double zero(double x, double epsilon) {
		return equal(x, 0.0, epsilon);
	}
	
	/**
	 * Defines the constraint x == 0.
	 * 
	 * @param x the first value
	 * @return the constraint value
	 */
	public static double zero(double x) {
		return zero(x, 0.0);
	}

	/**
	 * Defines the constraint x >= 0.
	 * 
	 * @param x the first value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double nonNegative(double x, double epsilon) {
		return greaterThanOrEqual(x, 0.0, epsilon);
	}

	/**
	 * Defines the constraint x >= 0.
	 * 
	 * @param x the first value
	 * @return the constraint value
	 */
	public static double nonNegative(double x) {
		return nonNegative(x, 0.0);
	}
	
	/**
	 * Defines the constraint x > 0.
	 * 
	 * @param x the first value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double positive(double x, double epsilon) {
		return greaterThan(x, 0.0, epsilon);
	}

	/**
	 * Defines the constraint x > 0.
	 * 
	 * @param x the first value
	 * @return the constraint value
	 */
	public static double positive(double x) {
		return positive(x, 0.0);
	}
	
	/**
	 * Defines the constraint x < 0.
	 * 
	 * @param x the first value
	 * @param epsilon the allowable error for the condition to still hold
	 * @return the constraint value
	 */
	public static double negative(double x, double epsilon) {
		return lessThan(x, 0.0, epsilon);
	}

	/**
	 * Defines the constraint x < 0.
	 * 
	 * @param x the first value
	 * @return the constraint value
	 */
	public static double negative(double x) {
		return negative(x, 0.0);
	}
	
	/**
	 * Requires all conditions to be satisfied.
	 * 
	 * @param args the individual constraint values
	 * @return the resulting constraint value
	 */
	public static double all(double... args) {
		double result = 0.0;
		
		for (int i = 0; i < args.length; i++) {
			result += args[i];
		}
		
		return result;
	}

	/**
	 * Requires at least one condition to be satisfied.
	 * 
	 * @param args the individual constraint values
	 * @return the resulting constraint value
	 */
	public static double any(double... args) {
		double result = 0.0;
		
		for (int i = 0; i < args.length; i++) {
			if (i == 0.0) {
				return 0.0;
			} else {
				result += args[i];
			}
		}
		
		return result;
	}

}
