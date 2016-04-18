package regression

import java.io.File

import breeze.linalg._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfter, FunSuite}

/**
 * Created by domesc on 03/04/16.
 */
@RunWith(classOf[JUnitRunner])
class LinearModelTest extends FunSuite with BeforeAndAfter{
  var data: DenseMatrix[Double] = _
  val model: LinearModel =  new LinearModel()

  before {
    val resourcesPath = System.getProperty("user.dir") + "/src/test/resources"
    data = csvread(new File(resourcesPath + "/data/test1.csv"))
  }

  test("Compute cost test") {
    val y: DenseVector[Double] = data(::, 1)
    val m: Int = y.length

    val X: DenseMatrix[Double] = DenseMatrix(DenseVector.ones[Double](m), data(::, 0))
    val theta: DenseVector[Double] = DenseVector.zeros(2)


    val cost = model.cost(X.t, y, theta)
    assert(Math.abs(cost - 32.07) <= 1.0e-2)
  }

  test("Gradient descent test") {
    val x: DenseMatrix[Double] = DenseMatrix((3.0, 1.0), (-1.0, -2.0))
    val y: DenseVector[Double] = DenseVector(5.4, 2.3)
    val theta: DenseVector[Double] = DenseVector.ones[Double](2)
    val alpha: Double = 0.01

    val (newTheta, history) = model.gradientDescent(x, y, theta, alpha, 1)

    val diff = newTheta - DenseVector(0.99, 0.97)
    diff.foreach(el => assert(el <= 1.0e-2))
  }




}
