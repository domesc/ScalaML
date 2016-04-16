package regression

import breeze.linalg._

/**
 * Created by domesc on 03/04/16.
 */
class LinearModel{

  def gradientDescent(X: DenseMatrix[Double],
                      y: DenseMatrix[Double],
                      theta: DenseMatrix[Double],
                      alpha: Double,
                      num_iters: Int): (DenseMatrix[Double], DenseVector[Double])= {
    val m: Int = y.rows
    val history: DenseVector[Double] = DenseVector.zeros(num_iters)

    @annotation.tailrec
    def descend(theta: DenseMatrix[Double],
                history: DenseVector[Double],
                remaining: Int): (DenseMatrix[Double], DenseVector[Double]) = remaining match {

      case 0 => (theta, history)
      case _ => {
        val updatedCost = alpha * (1/m) * (((X * theta) - y).t * X).t;
        val updatedTheta = theta - updatedCost
        history(remaining) = cost(X, y, updatedTheta);
        descend(updatedTheta, history, remaining - 1)
      }
    }

    descend(theta, history, num_iters)
  }

  def cost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double]): Double = {
    val m: Int = y.rows
    sum((X * theta - y) :^ 2d) / (2 * m)
  }

}
