package regression

import breeze.linalg._

/**
 * Created by domesc on 03/04/16.
 */
class LinearModel{

  /**
    * Gradient descent algorithm
    * @param X features
    * @param y labels
    * @param theta coefficients
    * @param alpha learning rate
    * @param num_iters number of iterations
    * @param lambda regularization parameter
    * @return cost
    */
  def gradientDescent(X: DenseMatrix[Double],
                      y: DenseVector[Double],
                      theta: DenseVector[Double],
                      alpha: Double,
                      num_iters: Int,
                      lambda: Double = 0.0): (DenseVector[Double], DenseVector[Double])= {
    val m: Int = y.length
    val history: DenseVector[Double] = DenseVector.zeros(num_iters)

    @annotation.tailrec
    def descend(theta: DenseVector[Double],
                history: DenseVector[Double],
                remaining: Int): (DenseVector[Double], DenseVector[Double]) = remaining match {

      case 0 => (theta, history)
      case _ => {
        val updatedCost = (alpha / m) * (((X * theta) - y).t * X).t;
        val updatedTheta = theta * (1 - (alpha * lambda) / m) - updatedCost
        history(remaining-1) = cost(X, y, updatedTheta, lambda);
        descend(updatedTheta, history, remaining - 1)
      }
    }

    descend(theta, history, num_iters)
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
