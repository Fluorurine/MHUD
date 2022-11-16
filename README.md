<h2> File chính nằm trong ./mainmhud.py </h2>
<h3>Dữ liệu gồm 2000 phần tử, 20 thuộc tính</h3>
<p>Model được train từ train.csv. Dự đoán từ test.csv và lưu kết quả vào Ketqua.csv</p>
<p>
<p><b>Các thuộc tính đặc trưng là :</b></p>
<i>
<ol>
<li>battery_power: Thời lượng pin (int)</li>
<li>blue: Thiết bị có kết nối được BlueTooth hay không (2 nhãn: 0,1)</li>
<li>clock_speed: Tốc độ xung đồng hồ (float)</li>
<li>dual_sim: Hỗ trợ sử dụng 2 sim (2 nhãn: 0,1)</li>
<li>fc : Độ phân giải camera trước (Megapixel) (int)</li>
<li>four_g: Hỗ trợ 4G (2 nhãn: 0,1)</li>
<li>int_memory: Bộ nhớ trong (GB) (int)</li>
<li>m_dep: Độ dày màn hình (cm) (float)</li>
<li>mobile_wt: Cân nặng (g) (int)</li>
<li>n_cores: Số lượng nhân (int)</li>
<li>pc: Độ phân giải camera chính (Megapixel) (int) </li>
<li>px_height: Độ phân giải màn hình ngang (int)</li>
<li>px_width: Độ phân giải màn hình dọc (int)</li>
<li>ram: Bộ nhớ trong (MB) (int)</li>
<li>sc_h: Chiều rộng màn hình (int)</li>
<li>sc_w: Chiều dài màn hình (int)</li>
<li>talk_time: Thời lượng sạc tối đa (int)</li>
<li>three_g: Hỗ trợ 3G (2 nhãn: 0,1)</li>
<li>touch_screen: Có chạm cảm ứng hay không (2 nhãn: 0,1)</li>
<li>wifi: Hỗ trợ kết nối wifi (2 nhãn: 0,1)</li>
</ol>
</i>
<p><b><u>Nhãn là cột price_range</u></b></p>
<p>
<b>Có 4 nhãn là:</b> 0, 1, 2, 3
</p>
<p>
<b>Số lượng:</b>  500 nhãn 0, 500 nhãn 1, 500 nhãn 2, 500 nhãn 3
</p>
