package org.borgmoea;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * A Pareto optimal set (the output of the Borg MOEA).
 */
public class Result implements Iterable<Solution> {
	
	/**
	 * A reference to the underlying C archive.
	 */
	private BorgLibrary.BORG_Archive archive;
	
	/**
	 * Optional statistics collected during the run.
	 */
	private List<Map<String, Double>> statistics;
	
	/**
	 * Constructs a new Pareto optimal set.
	 * 
	 * @param archive a reference to the underlying C archive
	 */
	Result(BorgLibrary.BORG_Archive archive) {
		this(archive, null);
	}
	
	/**
	 * Constructs a new Pareto optimal set.
	 * 
	 * @param archive a reference to the underlying C archive
	 * @param statistics statistics collected during the run
	 */
	Result(BorgLibrary.BORG_Archive archive, List<Map<String, Double>> statistics) {
		super();
		this.archive = archive;
		this.statistics = statistics;
	}

	@Override
	protected void finalize() throws Throwable {
		BorgLibrary.BORG_Archive_destroy(archive);
		super.finalize();
	}

	@Override
	public Iterator<Solution> iterator() {
		return new ResultIterator();
	}
	
	/**
	 * Returns {@code true} if statistics were collected during the run; {@code false} otherwise.
	 * 
	 * @return {@code true} if statistics were collected during the run; {@code false} otherwise
	 */
	public boolean hasStatistics() {
		return statistics != null;
	}
	
	/**
	 * Returns the statistics collected during the run, or {@code null} if no statistics are
	 * available.
	 * 
	 * @return the statistics collected during the run, or {@code null} if no statistics are
	 *         available
	 */
	public List<Map<String, Double>> getStatistics() {
		return statistics;
	}
	
	/**
	 * Returns the size of the Pareto optimal set.
	 * 
	 * @return the size of the Pareto optimal set
	 */
	public int size() {
		return BorgLibrary.BORG_Archive_get_size(archive);
	}
	
	/**
	 * Returns the Pareto optimal solution at the given index.
	 * 
	 * @param index the index
	 * @return the Pareto optimal solution at the given index
	 * @throws IndexOutOfBoundsException if the index is not valid
	 */
	public Solution get(int index) {
		if ((index < 0) || (index >= size())) {
			throw new IndexOutOfBoundsException();
		} else {
			return new Solution(BorgLibrary.BORG_Archive_get(archive, index));
		}
	}
	
	/**
	 * Iterates over the solutions in a Pareto optimal set.
	 */
	private class ResultIterator implements Iterator<Solution> {
		
		/**
		 * The current index.
		 */
		private int index;
		
		/**
		 * Constructs an iterator over the given Pareto optimal set.
		 */
		ResultIterator() {
			super();
			this.index = -1;
		}

		@Override
		public boolean hasNext() {
			return index < size()-1;
		}

		@Override
		public Solution next() {
			if (hasNext()) {
				index++;
				return get(index);
			} else {
				throw new NoSuchElementException();
			}
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
		
	}
	
}
