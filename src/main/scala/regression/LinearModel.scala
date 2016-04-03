package regression

import breeze.linalg._

/**
 * Created by domesc on 03/04/16.
 */
class LinearModel {

  def cost(X: DenseMatrix[Double], y: DenseMatrix[Double], theta: DenseMatrix[Double]): Double = {

    val m: Int = y.rows
    sum((X * theta - y) :^ 2d) / (2 * m)
  }

}
