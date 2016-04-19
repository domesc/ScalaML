package regression

import breeze.linalg._

/**
  * @param X features
  * @param y labels
  *
  * Created by domesc on 03/04/16.
  */
class LinearRegression(X: DenseMatrix[Double],
                       y: DenseVector[Double]) extends BaseModel{

  /**
    * Gradient descent algorithm
    * @param alpha learning rate
    * @param num_iters number of iterations
    * @param lambda regularization parameter
    * @return coefficients, costHistory
    */
  override def fit(alpha: Double,
                   num_iters: Int,
                   lambda: Double = 0.0): (DenseVector[Double], DenseVector[Double]) = {
    val m: Int = y.length
    val costHistory: DenseVector[Double] = DenseVector.zeros(num_iters)
    val theta = DenseVector.ones[Double](X.cols)

    @annotation.tailrec
    def descend(theta: DenseVector[Double],
                history: DenseVector[Double],
                remaining: Int): (DenseVector[Double], DenseVector[Double]) = remaining match {

      case 0 => (theta, history)
      case _ => {
        val updatedCost = (alpha / m) * (((X * theta) - y).t * X).t;
        val regularizationTerm = (1 - (alpha * lambda) / m)
        val updatedTheta = theta * regularizationTerm - updatedCost
        history(remaining-1) = cost(X, y, updatedTheta, lambda);
        descend(updatedTheta, history, remaining - 1)
      }
    }

    descend(theta, costHistory, num_iters)
  }

  /**
    * Cost function for linear regression model
    * @param X features
    * @param y labels
    * @param theta coefficients
    * @param lambda regularization parameter
    * @return cost
    */
  def cost(X: DenseMatrix[Double],
           y: DenseVector[Double],
           theta: DenseVector[Double],
           lambda: Double): Double = {
    val m: Int = y.length
    val regularizationTerm = lambda * sum(theta :^ 2d)
    sum((X * theta - y) :^ 2d) / (2 * m) + regularizationTerm
  }
}
