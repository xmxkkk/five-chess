/*
 Navicat Premium Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 50722
 Source Host           : localhost
 Source Database       : five

 Target Server Type    : MySQL
 Target Server Version : 50722
 File Encoding         : utf-8

 Date: 07/19/2018 18:49:41 PM
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `chess`
-- ----------------------------
DROP TABLE IF EXISTS `chess`;
CREATE TABLE `chess` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `steps` text,
  `md5` varchar(40) DEFAULT NULL,
  `create_time` datetime DEFAULT NULL,
  `winner` int(11) DEFAULT NULL,
  `step_num` int(11) DEFAULT NULL,
  `num` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1851 DEFAULT CHARSET=utf8mb4;

-- ----------------------------
--  Table structure for `step`
-- ----------------------------
DROP TABLE IF EXISTS `step`;
CREATE TABLE `step` (
  `chess_id` int(11) DEFAULT NULL,
  `board` text,
  `md5` varchar(36) DEFAULT NULL,
  `score` double(20,8) DEFAULT NULL,
  `idx` int(11) DEFAULT NULL,
  `create_time` datetime DEFAULT NULL,
  `step_num` int(11) DEFAULT NULL,
  `learn_num` int(11) DEFAULT NULL,
  `step_pos` varchar(10) DEFAULT NULL,
  `add_shape` varchar(255) DEFAULT NULL,
  `add_score` double(20,8) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SET FOREIGN_KEY_CHECKS = 1;
