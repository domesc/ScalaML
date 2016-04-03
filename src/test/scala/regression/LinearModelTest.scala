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

  before {
    val resourcesPath = System.getProperty("user.dir") + "/src/test/resources"
    data = csvread(new File(resourcesPath + "/data/test1.csv"))
  }

  test("Simple test") {
    val y: DenseVector[Double] = data(::, 1)
    val m: Int = y.length

    val X: DenseMatrix[Double] = DenseMatrix(DenseVector.ones[Double](m), data(::, 0))
    val theta: DenseVector[Double] = DenseVector.zeros(2)

    val model = new LinearModel()
    val cost = model.cost(X.t, y.toDenseMatrix.t, theta.toDenseMatrix.t)
    assert(BigDecimal(cost).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble == 32.07)
  }




}
