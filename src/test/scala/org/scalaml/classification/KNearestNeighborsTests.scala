package org.scalaml.classification

import breeze.linalg.DenseVector
import org.scalatest.{ FlatSpec, Matchers }

/**
 * Created by domesc on 20/11/16.
 */
class KNearestNeighborsTests extends FlatSpec with Matchers {

  "predictSample" should "give back the right class" in {
    val list: IndexedSeq[Double] = IndexedSeq(1.0, 2.0, 5.5, 5.5, 2.0, 2.0)
    val label = SimpleNearestNeighbors.predictSample(list)

    label shouldEqual 2.0
  }

  "predictSample" should "give back the class with highest value if multiple classes have same importance" in {
    val list: IndexedSeq[Double] = IndexedSeq(1.0, 2.0, 5.5, 5.5, 2.0)
    val label = SimpleNearestNeighbors.predictSample(list)

    label shouldEqual 5.5
  }

  "findTopK" should "find the top 2 classes with smaller distance" in {
    val row1 = DenseVector(1.0, 2.5, 3.1)
    val row2 = DenseVector(1.1, 5.5, 7.1)
    val row3 = DenseVector(8.9, 4.0, 3.2)
    val row4 = DenseVector(2.6, 10.0, 45.1)
    val features = IndexedSeq(row1, row2, row3, row4)
    val labels: IndexedSeq[Double] = IndexedSeq(1.0, 2.0, 1.0, 3.0)
    val model = SimpleNearestNeighbors(features, labels, k = 2)
    val sampleTest = DenseVector(1.2, 2.3, 5.2)
    val predictions = model.findTopK(sampleTest)

    predictions shouldEqual IndexedSeq(1.0, 2.0)
  }

  "predict" should "classify well" in {
    val row1 = DenseVector(1.0, 2.5, 3.1)
    val row2 = DenseVector(1.1, 5.5, 7.1)
    val row3 = DenseVector(8.9, 4.0, 3.2)
    val row4 = DenseVector(2.6, 10.0, 45.1)
    val row5 = DenseVector(1.1, 5.5, 7.1)
    val features = IndexedSeq(row1, row2, row3, row4, row5)
    val labels: IndexedSeq[Double] = IndexedSeq(1.0, 2.0, 1.0, 3.0, 2.0)
    val model = SimpleNearestNeighbors(features, labels, k = 3)
    val testSet: IndexedSeq[DenseVector[Double]] = IndexedSeq(DenseVector(1.2, 2.3, 5.2), DenseVector(8.9, 1.2, 40.0))
    val predictions = model.predict(testSet)

    predictions shouldEqual IndexedSeq(2.0, 2.0)
  }

}
