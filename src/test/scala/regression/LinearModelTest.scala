package regression

import java.io.File

import breeze.linalg._
import breeze.numerics.abs
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfter, FunSuite, Matchers}

/**
 * Created by domesc on 03/04/16.
 */
@RunWith(classOf[JUnitRunner])
class LinearModelTest extends FunSuite
  with Matchers
  with BeforeAndAfter{

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

    val cost = model.cost(X.t, y, theta, 0.0)
    assert(Math.abs(cost - 32.07) <= 1.0e-2)
  }

  test("Gradient descent test (with and without regularization)") {
    val X: DenseMatrix[Double] = DenseMatrix((3.0, 1.0), (-1.0, -2.0), (6.3, 5.2))
    val y: DenseVector[Double] = DenseVector(5.4, 2.3, 4.5)
    val theta: DenseVector[Double] = DenseVector.ones[Double](2)
    val alpha: Double = 0.01

    // verify normal fitting
    val (thetaNoReg, histNoReg) = model.gradientDescent(X, y, theta, alpha, 5000)
    val diffNoReg = abs(X * thetaNoReg - y)

    // check convergence and distance from target value
    histNoReg(0) shouldBe < (histNoReg(-1))
    diffNoReg.foreach(el => assert(el <= 2.5e-1))

    // fitting with regularization term
    val (thetaReg, histReg) = model.gradientDescent(X, y, theta, alpha, 5000, 1)
    val diffReg = abs(X * thetaNoReg - y)

    // check convergence and distance from target value
    histReg(0) shouldBe < (histReg(-1))
    diffReg.foreach(el => assert(el <= 2.5e-1))
  }




}
