package org.scalaml.api

import breeze.linalg.{ DenseMatrix, DenseVector }

import scala.collection.{ GenMap, GenSeq }

/**
 * Created by domesc on 29/10/16.
 */
trait UnsupervisedBaseModel[T] {

  /**
   * Create the [[GenMap]] with as key the centroid and as value the set of samples belonging to the centroid
   */
  def predict(): GenMap[T, GenSeq[T]]

}
