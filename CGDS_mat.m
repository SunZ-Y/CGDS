clear;
clc;
close;


file_path = '/Gait/hid/M_7'; % 原始文件保存路径
folder_path = '/Gait/H/h_n'; % 将路径替换为你想要保存的文件夹路径
folder_path1 = '/Gait/hid_pig/h_n'; % 将路径替换为你想要保存的文件夹路径
folder_path2 = '/Gait/hid_long/h_n'; % 将路径替换为你想要保存的文件夹路径



files_wen = dir(fullfile(file_path, '*.bin')); 
zongshu = length(files_wen);
disp(['bin文件数量为: ', num2str(zongshu)]);
    %% 输入数据
for file_num = 1:length(files_wen)
    geshu = file_num;
    disp(['目前bin文件为: ', num2str(geshu)]);
    filename = fullfile(file_path, files_wen(file_num).name);


    [A] = readDCA1000_1(filename); % 接受到的信号
    

    retVal = A(1,:);
    t0 = cputime;
    data_length  = length(retVal);
    desired_length = floor(data_length / 512 /32) * 512 * 32;
    retVal_1 = zeros(1,desired_length);
    for ii = 1:desired_length
        retVal_1(1,ii) = retVal(1,ii);
    end

    single = reshape(retVal_1,[512,desired_length/512]);
    single = single.';

    Fs=1.5e7;
    RampRate = 104.5e6/1e-6; %雷达调频斜率
    L = 32; % 一帧图像的chirp数
    N = 512; % 采样点
    k = (desired_length/512-480)/32; % 帧数
    Nf = 512; % 每帧样本数
    Nov = 480;% 每帧重叠样本数
    T = 0.01152;% 每帧时间
    B = 3.6e9; % 带宽
    C = 3*10^8; % 光速
    NRep = desired_length/512;   
    NFastFFT = 512; %快时间维度上FFT的升采样点数

    %% --------人体追踪---------
    %% 数据前处理

    % 对所有脉冲进行距离向成像

    RangeImage = zeros(NRep, NFastFFT);
    for ii = 1:NRep
        Single = squeeze(single(ii,:));
        ImgTemp = fft(Single, NFastFFT);
        RangeImage(ii, :) = ImgTemp;
    end
    toc
    tic
    Y = RangeImage.';
  
    % 数据傅里叶变换后的方差ZY
    Y = Y.';
    ZY = zeros(k, NFastFFT);  % 初始化方差矩阵
    ZX = zeros(Nf, NFastFFT);  %每一帧矩阵
    DeltaD = 1/N*Fs/RampRate/2*C;
    for ii = 1:k
        % 使用reshape函数替换循环
        start_idx = (ii-1)*L + 1;
        end_idx = (ii-1)*L + Nf;
        ZX = reshape(Y(start_idx:end_idx, :), Nf, NFastFFT);
        ZY(ii,:) = var(ZX, 1);
    end
    
    ZY = ZY.';
    %最大方差轨迹
    ZY_over = ZY.';
    ImgTemp = zeros(1, NFastFFT);
    MaxIndex = zeros(k, 1); %储存峰值位置下标

    for ii = 1:k
        ImgTemp = squeeze(ZY_over(ii,:)); 
        AbsImgTemp = abs(ImgTemp);
        [Waste, MaxIndex(ii)] = max(AbsImgTemp);
    end

    ZY_max = zeros(k, 1);% 最大方差
    for ii = 1:k
        ZY_max(ii,1) = ZY_over(ii, MaxIndex(ii));
    end
    ZY_logmax = log(ZY_max);
    toc
    tic
    %% 存在检测

    % 平滑
    window_size = 8; % 定义移动平均窗口的大小
    smoothed_data = movmean(ZY_logmax, window_size); % 计算移动平均
    % 阈值检测
    kernel_size = 3;  % 中值滤波器的核大小
    filtered_data = medfilt1(smoothed_data, kernel_size);% 应用中值滤波器
    threshold = 15;% 设置阈值
    above_threshold = filtered_data(filtered_data > threshold);% 根据阈值筛选数据
    below_threshold = filtered_data(filtered_data <= threshold);

    % 确定存在的数据
    AboveIndex = find(filtered_data > threshold);%存在数据-时间的角标
    N_Over = size(AboveIndex);
    n_Over = N_Over(1, 1);
    FFT_Index = zeros(k, 1);
    for ii = 1:n_Over
        FFT_Index(AboveIndex(ii,1),1) = 1;
    end
    % 减少间隔的检测误差
    for ii = 4:k-3
        if (sum(FFT_Index((ii-3):(ii-1),1) == 1) + sum(FFT_Index((ii+1):(ii+3),1) == 1)) >= 4 
            FFT_Index(ii,1) = 1;
        else
            FFT_Index(ii,1) = FFT_Index(ii,1);
        end
    end


    % 遍历数据
    for ii = 1:length(FFT_Index)
        % 判断前一百个数据或者后一百个数据是否连续等于1
        if ii >= 101 && sum(FFT_Index(ii-100:ii-1)) == 100
            FFT_Index(ii) = FFT_Index(ii);
        elseif ii <= length(FFT_Index)-100 && sum(FFT_Index(ii+1:ii+100)) == 100
            FFT_Index(ii) = FFT_Index(ii);
        else
            FFT_Index(ii) = 0;
        end
    end
    FFT_Index = FFT_Index.';

    % 首先找到从0到1的边界（即1的起始位置）
    start_indices = find(diff([0 FFT_Index]) == 1);

    % 然后找到从1到0的边界（即1的结束位置）
    end_indices = find(diff([FFT_Index 0]) == -1);

    % 计算每个连续的1段落的长度
    segment_lengths = end_indices - start_indices + 1;
    n10 = 0;
    N10 = zeros(1,length(segment_lengths)-1);
    for ii = 1:length(segment_lengths)-1
        N10(ii) = n10 + segment_lengths(ii);
        n10 = n10 + segment_lengths(ii);
    end

    FFT_Index = FFT_Index.';

    Index = find(FFT_Index==0);
    ZY_Live = ZY.';
    ZY_Live(Index, :) = [];

    % 输出删除后的ZY矩阵行数
    disp(['删除后的ZY矩阵行数：', num2str(size(ZY_Live, 1))]);
    N_ZYL = size(ZY_Live, 1);

    ZY_Live = ZY_Live.';
    toc
    %% 峰值跟踪

    if N_ZYL ~= 0
%         toc
        tic
        [sum, p] = calculateMaxSumWithCoordinates(ZY_Live); % 动态规划计算
        toc
        % 确定p*轨迹
        p_star = zeros(1, N_ZYL);
        p_star(1, :) = p(1, :)*2*DeltaD;
        % 取方差峰值对比
        Index_1 = zeros(1, N_ZYL);% 取方差峰值
        ZY_f= zeros(1, N_ZYL);
        for ii = 1:N_ZYL
            for jj = 1:NFastFFT
                if ZY_Live(jj, ii) == max(ZY_Live(:,ii))
                    Index_1(1,ii) = jj;
                    ZY_f(ii) = ZY_Live(jj, ii);
                end
            end
        end

        p_star1 = zeros(1, N_ZYL);
        p_star1(1, :) = Index_1(1, :)*2*DeltaD;
        toc
        tic
        %% 行走检测

        % 估计速度
        D = 100;
        v = zeros(1,N_ZYL);% 速度
        for ii = (D+1):N_ZYL
            v(1,ii) = (p_star(1,ii) - p_star(1,ii-D)) / (D * T);
        end
        % 估计状态
        m = zeros(1,N_ZYL);% 判断运动状态
        for ii = 2:N_ZYL
            if (abs(v(1,ii)) >= 0.4)&&(abs(v(1,ii)) <= 4)% 行走
                m(1,ii) = 1;
            elseif (abs(v(1,ii)) <= 0.2)||(abs(v(1,ii)) >= 4.5)% 静止
                m(1,ii) = 0;
            else
                m(1,ii) = m(1,ii-1);% 与前一状态相同
            end
        end

        nn = m;
        for ii = 1:length(N10)
            nn(N10(ii))  = 0;
        end

        % 首先找到从0到1的边界（即1的起始位置）
        start_indices = find(diff([0 nn]) == 1);

        % 然后找到从1到0的边界（即1的结束位置）
        end_indices = find(diff([nn 0]) == -1);

        % 计算每个连续的1段落的长度
        segment_lengths = end_indices - start_indices + 1;
        nn10 = 0;
        NN10 = zeros(1,length(segment_lengths)-1);
        for ii = 1:length(segment_lengths)-1
            NN10(ii) = nn10 + segment_lengths(ii);
            nn10 = nn10 + segment_lengths(ii);
        end

        non_zero_v = (m ~= 0);% 创建逻辑向量，指示非零元素的位置
        v_1 = v(non_zero_v);% 使用逻辑索引选择非零元素
        Z_m = ZY_f(non_zero_v);% 方差峰值
        toc
        tic

        %% --------GaitCube图形提取----------
        %% 谱图提取

        % 确定数据
        jj = 0;
        for ii = 1:k
            if FFT_Index(ii) == 1
                jj = jj + 1;
                FFT_Index(ii) = m(jj);% 确定数据的角标
            end
        end

        Index_walk = find(m==0);
        ZY_Live(:, Index_walk) = [];% 行走的方差

        p_walk = p(1,:);
        p_walk(:,Index_walk) = []; 

        % 速度-多普勒
        N_Body = 9;
        r = 2*N_Body +1;
        F_w = size(p_walk,2);% 行走状态的帧数

        Y_F = zeros(F_w, N,r);
        jj = 0;
        for ii = 1:k
            if FFT_Index(ii)==1
                jj = jj+1;
                if (p_walk(jj)>=N_Body+1) && (p_walk(jj)<=NFastFFT-N_Body)
                    Y_F(jj,:,:) = Y(((ii-1)*L+1):(ii+15)*L,(p_walk(jj)-N_Body):(p_walk(jj)+N_Body));
                end
            end
        end

        YD1 = zeros(F_w, N, r);
        sigDwin=zeros(N,r);
        hamming2=hamming(N);
        for frame = 1:F_w
                for range_bin = 1:r
                    sigDwin(:, range_bin) = Y_F(frame, :, range_bin);
                    sigDwin(:, range_bin) = hamming2.*sigDwin(:, range_bin);
                    YD1(frame,:, range_bin) = (abs(fftshift(fft(squeeze(sigDwin(:, range_bin)), N)))).^2;% 距离方向的变换
                end
        end

        % 绘制距离多普勒图
        XAxis = linspace(0, (F_w-1)*T,  F_w);
        doppler_axis = linspace(-Fs/4, Fs/4, 2*Nf);
        % YD_Y = abs(YD1(:,:,N_Body+1));


        Y_D = YD1(:,:,1).';
        for ii = 2:r
            Y_D = Y_D + YD1(:,:,ii).';
        end
       toc;
        %% 步态周期估计

       % 周期提取

        spectrum_energy = log(abs(Y_D).^2);% 计算频谱能量

        mean_log_energy = mean(abs(spectrum_energy), 1);% 计算每个时间帧的平均对数能量
        windowSize = 75;
        % 计算移动平均
        trend = movmean(mean_log_energy, windowSize);
        % 去除趋势
        detrended = mean_log_energy - trend;
        
        smoothed_data = smoothdata(detrended, 'loess'); % 使用 LOESS 平滑滤波器
        smoothed_data = smoothdata(smoothed_data, 'loess'); % 再次使用 LOESS 平滑滤波器
        smoothed_data = smoothdata(smoothed_data, 'loess'); % 再次使用 LOESS 平滑滤波器
       
        
        % 步态切割
        S = smoothed_data;
        jj = 1;
        S_Index = zeros();

        for ii = 1:15
            if S(ii) == min(S(1:15))
                S_Index(jj) = ii; 
                jj = jj + 1;
            end
        end
        for ii = 16:F_w-15
            if S(ii)<=min(S(ii-15:ii-1)) && S(ii)<=min(S(ii+1:ii+15))
                S_Index(jj) = ii; 
                jj = jj + 1;
            end
        end
        for ii = F_w-15:F_w
            if S(ii) == min(S(F_w-15:F_w))
                S_Index(jj) = ii; 

            end
        end




        S_Size = size(S_Index,2);% 满足条件的最小点
        jj = 1;
        S_Index_0 = zeros(1,S_Size); 
        for ii = 1:S_Size
            if (ii == 1) && ((S_Index(ii+1)-S_Index(ii))>=30)
                S_Index_0(jj) = S_Index(ii);
                jj = jj+1;
            elseif ((ii>=2) && (ii<=S_Size-1)) && (((S_Index(ii)-S_Index(ii-1))>=30)||((S_Index(ii+1)-S_Index(ii))>=30))
                S_Index_0(jj) = S_Index(ii);
                jj = jj+1;
            elseif (ii == S_Size) && ((S_Index(ii)-S_Index(ii-1))>=30)
                S_Index_0(jj) = S_Index(ii);
            end
        end

        non_zero_indices = (S_Index_0 ~= 0);% 创建逻辑向量，指示非零元素的位置
        S_Index_1 = S_Index_0(non_zero_indices);% 使用逻辑索引选择非零元素

        S_Size_1 = size(S_Index_1,2);% 去除相近的最小值
        for ii = 1:S_Size_1-1
            if (ii==1) && ((S_Index_1(ii+1)-S_Index_1(ii))<=30)
                S_Index_1(ii) = 0;
            elseif (S_Index_1(ii)~=0) && ((S_Index_1(ii+1)-S_Index_1(ii))<=30) 
                S_Index_1(ii+1) = 0;
            elseif (S_Index_1(ii) == 0) && ((S_Index_1(ii+1)-S_Index_1(ii-1))<=30)
                S_Index_1(ii+1) = 0;
            end
        end

        non_zero_indices = (S_Index_1 ~= 0);% 创建逻辑向量，指示非零元素的位置
        S_Index_1 = S_Index_1(non_zero_indices);% 使用逻辑索引选择非零元素



        % 获取文件夹中的所有文件信息
        files = dir(fullfile(folder_path, '*.*'));

        % 计算文件夹中文件的数量（不包括文件夹本身和上级目录）
        num_files = length(files) - 2 + 10000; % 减去 . 和 ..

        % 显示结果
        disp(['文件夹中的文件数量为: ', num2str(num_files)]);

        path_size = size(S_Index_1,2);
        YL = zeros(1,path_size);% 检测是否正确
        for ii = 1:path_size-1
               YL(ii) = S_Index_1(ii+1) - S_Index_1(ii);
        end


        jj =num_files + 1;
        add_1 = zeros(1,6);
        add_5 = zeros(1,6);
        ssss = size(S_Index_1,2);
        for ii = 1:5:path_size-5
            if ii < ssss - 4
                all_outside_range5 = all(NN10 < S_Index_1(ii) | NN10 > S_Index_1(ii+5));
            else
                all_outside_range5 = 0;
            end
            if all_outside_range5
                v_s5 = median(v_1(S_Index_1(ii):S_Index_1(ii+5)));
                add_5(1,4) = mean(v_1(S_Index_1(ii):S_Index_1(ii+5)));% 速度
                if (all(abs(v_1(S_Index_1(ii):S_Index_1(ii+5))) <= (10*abs(v_s5)))) && (all(abs(v_1(S_Index_1(ii):S_Index_1(ii+5)))) >= (0.1*abs(v_s5)))
                    if ((S_Index_1(ii+5)-S_Index_1(ii)) >= 100) && ((S_Index_1(ii+5)-S_Index_1(ii)) <= 800)
                        % 生成图片
                        % 构建文件名
                        add_5(1,1) =  median(Z_m(S_Index_1(ii):S_Index_1(ii+5)));% 方差
                        add_5(1,2) = (S_Index_1(ii+5)-S_Index_1(ii))*0.00512;% 时间
                        add_5(1,3) = p_walk(S_Index_1(ii+5))-p_walk(S_Index_1(ii));% 距离
                        heatmaps5 = zeros(227,495,5);



                        for kk = 2:4:18
                            f1 = figure;
                            YD_Y = medfilt2(abs(YD1(:,:,kk)),[3 3]);
                            YD_Y = YD_Y.';

                            gray_image5 = mat2gray(10*log10(abs(squeeze(YD_Y(:,S_Index_1(ii):S_Index_1(ii+5))))));  % 将矩阵归一化为 [0, 1] 的灰度图像

                            % 调整灰度图像的大小为 227x99
                            resized_gray_image5 = imresize(gray_image5, [227, 495]);

                            % 再由灰度图像得到 227x99 的矩阵（将灰度图像转换为矩阵）
                            matrix_from_gray_image5 = resized_gray_image5;
                            %                 max_matrix = max(matrix_from_gray_image(:));
                            %                 matrix_from_gray_image = matrix_from_gray_image / max_matrix ;
                            heatmaps5(:,:,(kk+2)/4) = matrix_from_gray_image5;

                            close(f1);
                        end
                        for iii = ii:ii+5
                            if iii < ssss
                                all_outside_range = all(NN10 < S_Index_1(iii) | NN10 > S_Index_1(iii+1));
                            else
                                all_outside_range = 0;
                            end
                            if all_outside_range
                                v_s = median(v_1(S_Index_1(iii):S_Index_1(iii+1)));
                                add_1(1,4) = mean(v_1(S_Index_1(iii):S_Index_1(iii+1)));% 速度
                                v_m = add_1(1,4);
                                if (all(abs(v_1(S_Index_1(iii):S_Index_1(iii+1))) <= (4*abs(v_s)))) && (all(abs(v_1(S_Index_1(iii):S_Index_1(iii+1)))) >= (0.25*abs(v_s)))
                                    if ((S_Index_1(iii+1)-S_Index_1(iii)) >= 30) && ((S_Index_1(iii+1)-S_Index_1(iii)) <= 100)
                                        % 生成图片
                                        % 构建文件名
                                        add_1(1,1) =  median(Z_m(S_Index_1(iii):S_Index_1(iii+1)));% 方差
                                        Z_mm = add_1(1,1);
                                        add_1(1,2) = (S_Index_1(iii+1)-S_Index_1(iii))*0.00512;% 时间
                                        tt = add_1(1,2);
                                        add_1(1,3) = p_walk(S_Index_1(iii+1))-p_walk(S_Index_1(iii));% 距离
                                        ss = add_1(1,3);
                                        heatmaps = zeros(227,99,5);

                                        % file_name = ['struct_', num2str(jj), '.mat']; % 在文件名中包含循环索引
                                        file_name = ['img_', num2str(jj), 'v_', num2str(v_m), 't_', num2str(tt),'s_', num2str(ss),'Z_', num2str(Z_mm),'.mat']; % 在文件名中包含循环索引
                                        for kkk = 2:4:18
                                            f2 = figure;
                                            YD_Y = medfilt2(abs(YD1(:,:,kkk)),[3 3]);
                                            YD_Y = YD_Y.';

                                            gray_image = mat2gray(10*log10(abs(squeeze(YD_Y(:,S_Index_1(iii):S_Index_1(iii+1))))));  % 将矩阵归一化为 [0, 1] 的灰度图像

                                            % 调整灰度图像的大小为 227x99
                                            resized_gray_image = imresize(gray_image, [227, 99]);

                                            % 再由灰度图像得到 227x99 的矩阵（将灰度图像转换为矩阵）
                                            matrix_from_gray_image = resized_gray_image;
                                            %                 max_matrix = max(matrix_from_gray_image(:));
                                            %                 matrix_from_gray_image = matrix_from_gray_image / max_matrix ;
                                            heatmaps(:,:,(kkk+2)/4) = matrix_from_gray_image;

                                            close(f2);
                                        end                                                         
                                        full_file_path = fullfile(folder_path, file_name);
                                        save(full_file_path,'heatmaps','heatmaps5','add_1','add_5');
                                        
                                        jj = jj+1;
                                     end
                                end
                            end
                        end
                    end
                end
            end
        end
        t1 = cputime - t0;
        disp([num2str(t1),'秒']);

        files1 = dir(fullfile(folder_path1, '*.*'));
        % 计算文件夹中文件的数量（不包括文件夹本身和上级目录）
        num_files1 = length(files1)-2 + 10000; % - 2; % 减去 . 和 ..
        % 显示结果
        disp(['5_long文件夹中的文件数量为: ', num2str(num_files1)]);

        files2 = dir(fullfile(folder_path2, '*.*'));
        % 计算文件夹中文件的数量（不包括文件夹本身和上级目录）
        num_files2 = length(files2)-2; %- 2; % 减去 . 和 ..
        % 显示结果
        disp(['long文件夹中的文件数量为: ', num2str(num_files2)]);
        path_size1 = size(S_Index_1,2);

        YL = zeros(1,path_size1);
        for ii = 1:path_size1-1
            YL(ii) = S_Index_1(ii+1) - S_Index_1(ii);
        end
        x = linspace(0, 1);
        jj1 =num_files2 + 1;
        jjj = num_files1 + 1;
        for ii = 1:5:path_size-5
            if ii < ssss - 4
                all_outside_range5 = all(NN10 < S_Index_1(ii) | NN10 > S_Index_1(ii+5));
            else
                all_outside_range5 = 0;
            end
            if all_outside_range5
                v_s5 = median(v_1(S_Index_1(ii):S_Index_1(ii+5)));
                v_m = mean(v_1(S_Index_1(ii):S_Index_1(ii+5)));
                if (all(abs(v_1(S_Index_1(ii):S_Index_1(ii+5))) <= (10*abs(v_s5)))) && (all(abs(v_1(S_Index_1(ii):S_Index_1(ii+5)))) >= (0.1*abs(v_s5)))
                    if ((S_Index_1(ii+5)-S_Index_1(ii)) >= 100) && ((S_Index_1(ii+5)-S_Index_1(ii)) <= 800)


                        f2 = figure;
                        colormap(jet);
                        imagesc(x, doppler_axis, 10*log10(abs(squeeze(Y_D(:,S_Index_1(ii):S_Index_1(ii+5))))));

                        % 构建文件名
                        file_name5 = ['img_', num2str(jj1), '.jpg']; % 在文件名中包含循环索引
                        jj1 = jj1+1;
                        % 拼接完整的文件路径
                        full_file_path = fullfile(folder_path2, file_name5);

                        % 保存图形到文件夹中
                        saveas(f2, full_file_path);

                        % 关闭当前图形以便下一次循环
                        close(f2);

                        for iii = ii:ii+5
                            if iii < ssss
                                all_outside_range = all(NN10 < S_Index_1(iii) | NN10 > S_Index_1(iii+1));
                            else
                                all_outside_range = 0;
                            end
                            if all_outside_range
                                v_s = median(v_1(S_Index_1(iii):S_Index_1(iii+1)));
                                add_1(1,4) = mean(v_1(S_Index_1(iii):S_Index_1(iii+1)));% 速度
                                if (all(abs(v_1(S_Index_1(iii):S_Index_1(iii+1))) <= (4*abs(v_s)))) && (all(abs(v_1(S_Index_1(iii):S_Index_1(iii+1)))) >= (0.25*abs(v_s)))
                                    if ((S_Index_1(iii+1)-S_Index_1(iii)) >= 30) && ((S_Index_1(iii+1)-S_Index_1(iii)) <= 100)
                                        f1 = figure;
                                        colormap(jet);
                                        imagesc(x, doppler_axis, 10*log10(abs(squeeze(Y_D(:,S_Index_1(iii):S_Index_1(iii+1))))));

                                        % 构建文件名
                                        file_name = ['img_',num2str(jjj) ,'ii_',num2str(jj1),'i_',num2str(iii-ii+1) ,'.jpg']; % 在文件名中包含循环索引
                                        jjj = jjj+1;
                                        % 拼接完整的文件路径
                                        full_file_path = fullfile(folder_path1, file_name);

                                        % 保存图形到文件夹中
                                        saveas(f1, full_file_path);

                                        % 关闭当前图形以便下一次循环
                                        close(f1);

                                        jj = jj+1;
                                     end
                                end
                            end
                        end
                    end
                end
             end
        end

        clearvars -except file_path files_wen file_num folder_path1 folder_path folder_path2;
        close all;
elseif N_ZYL == 0
    clearvars -except file_path files_wen file_num folder_path1 folder_path folder_path2;
    close all;
    end
end

