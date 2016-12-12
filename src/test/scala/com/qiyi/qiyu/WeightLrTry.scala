package com.qiyi.qiyu

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression

object WeightLrTry {
  val spark = SparkSession
      .builder()
      .appName("Weight LR test")
      .master("local")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

  import spark.implicits._

  case class Point(weight: Double, label: Double, features: Vector){
  }

  def getWeightLrData(filePath: String)(implicit sc: SparkContext) = {
    val dataRDD = sc.textFile(filePath)
    val parsed = dataRDD.map(_.split(" ")).map { parts =>
      val weight = parts(0).toDouble
      val label = parts(1).toDouble
      val (indices, values) = parts.slice(2, parts.length).filter(_.nonEmpty).map { p =>
        val indexAndValue = p.split(':')
        require(indexAndValue.length == 2, "LibSVM File format wrong")
        val index = indexAndValue(0).toInt
        val value = indexAndValue(1).toDouble
        (index, value)
      }.unzip
      (weight, label, indices.toArray, values.toArray)
    }
    parsed.persist(StorageLevel.MEMORY_ONLY)
    val d = parsed.map {
      case (weight: Double, label: Double, indices: Array[Int], values: Array[Double]) =>
        indices.lastOption.getOrElse(0)
    }.reduce(math.max)
    parsed.map { case(weight, label, indices, values) =>
      new Point(weight, label, Vectors.sparse(d, indices, values))
    }
  }

  def main(args: Array[String]): Unit = {

    implicit val sc = spark.sparkContext

    val filePath = "D://qiyi/online_data_sample"
    val dataFrame = getWeightLrData(filePath).toDF()
    val trainer1 = (new LogisticRegression).setFitIntercept(true).setStandardization(true)
      .setWeightCol("weight")

    val model1 = trainer1.fit(dataFrame)
  }
}
