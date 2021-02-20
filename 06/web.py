html='''
<!doctype html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
	<meta http-equiv="X-UA-Compatible" content="ie=edge">
	<title>五子棋网页版</title>
	<style>
		.wrapper {
	    width: 700px;
	    position: relative;
	    margin: 0 auto;
	    margin-bottom:0px;
        }

/* 棋盘 */
div.chessboard {
	margin: 0px 0 0 0;
	width: 542px;
	background: url(static/images/chessboard.png) no-repeat 14px 14px rgb(250, 250, 250);
	overflow: hidden;
	box-shadow: 2px 2px 8px #888;
	-webkit-box-shadow: 2px 2px 8px #888;
	-moz-box-shadow: 2px 2px 8px #888;
}

div.chessboard div {
	float: left;
	width: 36px;
	height: 36px;
	border-top: 0px solid #ccc;
	border-left: 0px solid #ccc;
	border-right: 0;
	border-bottom: 0;
	cursor: pointer;
}

/* 棋子 */
div.chessboard div.black {
	background: url(static/images/black.png) no-repeat 4px 4px;
}
div.chessboard div.white {
	background: url(static/images/white.png) no-repeat 4px 4px;
}
div.chessboard div.hover {
	background: url(static/images/hover.png) no-repeat 1px 1px;
}
div.chessboard div.hover-up {
	background: url(static/images/hover_up.png) no-repeat 1px 1px;
}
div.chessboard div.hover-down {
	background: url(static/images/hover_down.png) no-repeat 1px 1px;
}
div.chessboard div.hover-up-left {
	background: url(static/images/hover_up_left.png) no-repeat 1px 1px;
}
div.chessboard div.hover-up-right {
	background: url(static/images/hover_up_right.png) no-repeat 1px 1px;
}
div.chessboard div.hover-left {
	background: url(static/images/hover_left.png) no-repeat 1px 1px;
}
div.chessboard div.hover-right {
	background: url(static/images/hover_right.png) no-repeat 1px 1px;
}
div.chessboard div.hover-down-left {
	background: url(static/images/hover_down_left.png) no-repeat 1px 1px;
}
div.chessboard div.hover-down-right {
	background: url(static/images/hover_down_right.png) no-repeat 1px 1px;
}
div.chessboard div.white-last {
	background: url(static/images/white_last.png) no-repeat 4px 4px;
}
div.chessboard div.black-last {
	background: url(static/images/black_last.png) no-repeat 4px 4px;
}

/* 右侧 */

div.operating-panel {
	position: absolute;
	left: 550px;
	top: 150px;
	width: 200px;
	text-align: center;
}

.operating-panel a {
	display: inline-block;
	padding: 10px 15px;
	margin-bottom: 39px;
	margin-right: 8px;
	margin-left: 8px;
	background: rgb(100, 167, 233);
	text-decoration: none;
	color: #333;
	font-weight: bold;
	font-size: 16px;
	font-family: "微软雅黑", "宋体";
}

.operating-panel a:hover {
	background: rgb(48, 148, 247);
	text-decoration: none;
}

.operating-panel a.disable, .operating-panel a.disable:hover {
	cursor: default;
	background: rgb(197, 203, 209);
	color: rgb(130, 139, 148);
}

.operating-panel a.selected {
	border: 5px solid #F3C242;
	padding: 5px 10px;
}

#result_tips {
	color: #CE4242;
	font-size: 26px;
	font-family: "微软雅黑";
}

#result_info {
	margin-bottom: 26px;
}
#gengduo {
margin-top: 54px;
position: absolute;
    left: 620px;
    top: 450px;
    width: 180px;
    text-align: center;
}
#gengduo a {
    display: inline-block;
    padding: 10px 15px;
    margin-bottom: 39px;
    margin-right: 8px;
    margin-left: 8px;
    background: rgb(100, 167, 233);
    text-decoration: none;
    color: #333;
    font-weight: bold;
    font-size: 16px;
    font-family: "微软雅黑", "宋体";
}
#gengduo a:hover {
	background: rgb(48, 148, 247);
	text-decoration: none;
}
#footr {
text-align:center;
margin:28px auto;
    font-size: 12px;
    color: #000;
}
/* #footr a {
				border: 1px solid #2773bf;
				background-color: #358ae2;
				color: #fff;
				width: 148px;
				height: 38px;
				line-height: 38px;
				font-size: 17px;
				display: inline-block;
				text-align: center;
				cursor: pointer;
				text-decoration: none;
				margin-top: 20px;
			}
#footr a:hover{
				background: #2171c2;
			} */

</style>
</head>
<body>
<div class="wrapper">
	<div class="chessboard">
		<!-- top line -->
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top"></div>
		<div class="chess-top chess-right"></div>
		<!-- line 1 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 2 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 3 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 4 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 5 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 6 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 7 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 8 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 9 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 10 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 11 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 12 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- line 13 -->
		<div class="chess-left"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-middle"></div>
		<div class="chess-right"></div>
		<!-- bottom line  -->
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom"></div>
		<div class="chess-bottom chess-right"></div>
	</div>

	<div class="operating-panel">
		<p>
			<a id="black_btn" class="btn selected disable" href="javascript:;">黑&nbsp;&nbsp;子</a>
			<a id="white_btn" class="btn disable" href="javascript:;">白&nbsp;&nbsp;子</a>
		</p>
		<p>
			<a id="first_move_btn" class="btn selected disable" href="javascript:;">先&nbsp;&nbsp;手</a>
			<a id="second_move_btn" class="btn disable" href="javascript:;">后&nbsp;&nbsp;手</a>
		</p>
		<a id="replay_btn" class="btn" href="javascript:;">重&nbsp;&nbsp;&nbsp;玩</a>
		<p id="result_info">胜率：0%</p>
		<p id="result_tips"></p>
	</div>

	<div style="display: none">
		<!-- 图片需合并 减少http请求数 -->
		<img src="black.png" tppabs="http://www.cynking.com/static/images/black.png" alt="preload">
		<img src="white.png" tppabs="http://www.cynking.com/static/images/white.png" alt="preload">
		<img src="hover.png" tppabs="http://www.cynking.com/static/images/hover.png" alt="preload">
		<img src="hover_up.png" tppabs="http://www.cynking.com/static/images/hover_up.png" alt="preload">
		<img src="hover_down.png" tppabs="http://www.cynking.com/static/images/hover_down.png" alt="preload">
		<img src="hover_up_left.png" tppabs="http://www.cynking.com/static/images/hover_up_left.png" alt="preload">
		<img src="hover_up_right.png" tppabs="http://www.cynking.com/static/images/hover_up_right.png" alt="preload">
		<img src="hover_left.png" tppabs="http://www.cynking.com/static/images/hover_left.png" alt="preload">
		<img src="hover_right.png" tppabs="http://www.cynking.com/static/images/hover_right.png" alt="preload">
		<img src="hover_down_left.png" tppabs="http://www.cynking.com/static/images/hover_down_left.png" alt="preload">
		<img src="hover_down_right.png" tppabs="http://www.cynking.com/static/images/hover_down_right.png" alt="preload">
		<img src="black_last.png" tppabs="http://www.cynking.com/static/images/black_last.png" alt="preload">
		<img src="white_last.png" tppabs="http://www.cynking.com/static/images/white_last.png" alt="preload">
	</div>
</div>
<script src="jquery.min.js-v=1" tppabs="http://cynking.com/static/js/jquery.min.js?v=1"></script>
<script src="game_gobang.js-v=1" tppabs="http://cynking.com/static/js/game_gobang.js?v=1"></script>
</body>
</html>
'''
