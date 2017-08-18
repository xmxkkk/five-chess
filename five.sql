/*
 Navicat MySQL Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 50718
 Source Host           : localhost
 Source Database       : five

 Target Server Type    : MySQL
 Target Server Version : 50718
 File Encoding         : utf-8

 Date: 08/18/2017 18:33:24 PM
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `chess`
-- ----------------------------
DROP TABLE IF EXISTS `chess`;
CREATE TABLE `chess` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `steps` text,
  `md5` varchar(255) DEFAULT NULL,
  `create_time` datetime DEFAULT NULL,
  `winner` int(11) DEFAULT NULL,
  `num` int(11) NOT NULL DEFAULT '0',
  `learn_num` int(11) NOT NULL DEFAULT '0',
  `step_num` int(11) NOT NULL DEFAULT '0',
  `is_dev` int(11) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8249 DEFAULT CHARSET=utf8;

-- ----------------------------
--  Table structure for `step`
-- ----------------------------
DROP TABLE IF EXISTS `step`;
CREATE TABLE `step` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `chess_id` int(11) DEFAULT NULL,
  `board` text,
  `md5` varchar(255) DEFAULT NULL,
  `score` double(10,6) DEFAULT NULL,
  `create_time` datetime DEFAULT NULL,
  `idx` int(11) DEFAULT NULL,
  `step_num` int(11) NOT NULL DEFAULT '0',
  `learn_num` int(11) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=234761 DEFAULT CHARSET=utf8;

SET FOREIGN_KEY_CHECKS = 1;
