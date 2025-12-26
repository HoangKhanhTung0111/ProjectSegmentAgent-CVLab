import cv2
import sys
import os
import numpy as np

from modules.segmentation.intelligent_scissors import IntelligentScissorsApp
from modules.inpainting.strategies import TraditionalInpainting

def main():
    """
    Main application entry point.
    Workflow: Segmentation (Interactive) -> Inpainting (Automatic)
    """
    
    # =========================
    # SETUP & VALIDATION
    # =========================
    image_path = "inputs/test_image2.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy file '{image_path}'")
        sys.exit(1)
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc ảnh gốc
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Lỗi: Không thể đọc ảnh '{image_path}'")
        sys.exit(1)
    
    print("=" * 70)
    print("🎨 INTELLIGENT SCISSORS SEGMENTATION + TRADITIONAL INPAINTING")
    print("=" * 70)
    print(f"📁 Đã tải ảnh: {image_path}")
    
    # =================================================================
    # BƯỚC 1: SEGMENTATION (Tạo Mask) - SỬ DỤNG INTELLIGENT SCISSORS
    # =================================================================
    print("\n" + "=" * 70)
    print("📍 BƯỚC 1: SEGMENTATION - Tạo Mask bằng Intelligent Scissors")
    print("=" * 70)
    
    # Khởi tạo Interactive Segmentation Tool
    seg_app = IntelligentScissorsApp(image_path)
    
    # [THAY ĐỔI TỪ CODE CŨ]
    # Trước đây: seg_model = ColorBasedSegmentation(color_range='yellow')
    # Bây giờ:   Dùng IntelligentScissorsApp để người dùng tự vẽ mask
    
    print("\n📋 HƯỚNG DẪN VẼ MASK:")
    print("  🖱️  Chuột Trái  : Thêm điểm neo")
    print("  🖱️  Chuột Phải  : Kết thúc vòng vẽ (lưu vào mask)")
    print("  ⌨️  ENTER       : Kết thúc vòng vẽ")
    print("  ⌨️  BACKSPACE   : Undo bước trước")
    print("  ⌨️  ESC         : HOÀN TẤT SEGMENTATION và chuyển sang Inpainting")
    print("=" * 70 + "\n")
    
    # Chạy vòng lặp Interactive Segmentation
    seg_app.update_display()
    
    while True:
        key = cv2.waitKey(20) & 0xFF
        
        if key == 27:  # ESC - Hoàn tất Segmentation
            print("\n✅ Đã hoàn tất Segmentation!")
            break
        elif key == 13:  # ENTER - Kết thúc vòng vẽ
            if seg_app.is_started:
                seg_app.finish_drawing()
        elif key == 8:  # BACKSPACE - Undo
            seg_app.undo_last_step()
    
    # Lấy mask đã vẽ
    mask = seg_app.global_mask.copy()
    
    # Kiểm tra mask có rỗng không
    if cv2.countNonZero(mask) == 0:
        print("⚠️  Cảnh báo: Mask rỗng! Không có vùng nào được chọn.")
        print("💡 Bạn có thể:")
        print("   - Chạy lại và vẽ mask")
        print("   - Hoặc thoát nếu không cần xử lý")
        cv2.destroyAllWindows()
        
        response = input("\nBạn có muốn thoát không? (y/n): ")
        if response.lower() == 'y':
            sys.exit(0)
        else:
            # Chạy lại từ đầu
            cv2.destroyAllWindows()
            return main()
    
    # Lưu mask để debug/kiểm tra
    mask_path = os.path.join(output_dir, "01_segmentation_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"💾 Đã lưu Mask: {mask_path}")
    
    # Hiển thị Mask để kiểm tra
    cv2.imshow("Debug: Generated Mask", mask)
    print("\n👁️  Đang hiển thị mask... Nhấn phím bất kỳ để tiếp tục.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # =================================================================
    # BƯỚC 2: INPAINTING (Xóa vùng đã chọn)
    # =================================================================
    print("\n" + "=" * 70)
    print("🖌️  BƯỚC 2: INPAINTING - Xóa vùng đã chọn")
    print("=" * 70)
    
    # Khởi tạo Inpainting Strategy
    # Có thể chọn method='ns' (Navier-Stokes) hoặc 'telea'
    inpainting_model = TraditionalInpainting(method='ns', radius=3)
    
    print("🔄 Đang thực hiện Inpainting...")
    try:
        # Áp dụng inpainting
        inpainted_image = inpainting_model.process(original_image, mask)
        print("✅ Inpainting hoàn tất!")
        
    except Exception as e:
        print(f"❌ Lỗi khi Inpainting: {e}")
        sys.exit(1)
    
    # =================================================================
    # BƯỚC 3: HIỂN THỊ & LƯU KẾT QUẢ
    # =================================================================
    print("\n" + "=" * 70)
    print("📊 BƯỚC 3: HIỂN THỊ KẾT QUẢ")
    print("=" * 70)
    
    # Tạo ảnh so sánh Before/After
    comparison = np.hstack([original_image, inpainted_image])
    
    # Lưu các kết quả
    result_path = os.path.join(output_dir, "02_inpainted_result.png")
    comparison_path = os.path.join(output_dir, "03_comparison.png")
    
    cv2.imwrite(result_path, inpainted_image)
    cv2.imwrite(comparison_path, comparison)
    
    print(f"💾 Đã lưu ảnh kết quả: {result_path}")
    print(f"💾 Đã lưu ảnh so sánh: {comparison_path}")
    
    # Hiển thị kết quả
    cv2.imshow("Result: Before (Left) vs After (Right)", comparison)
    print("\n👁️  Đang hiển thị kết quả... Nhấn phím bất kỳ để thoát.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # =================================================================
    # HOÀN TẤT
    # =================================================================
    print("\n" + "=" * 70)
    print("✨ HOÀN THÀNH!")
    print("=" * 70)
    print(f"📁 Các file đã được lưu trong thư mục: {output_dir}/")
    print("   1. 01_segmentation_mask.png    - Mask đã vẽ")
    print("   2. 02_inpainted_result.png     - Ảnh sau inpainting")
    print("   3. 03_comparison.png           - Ảnh so sánh Before/After")
    print("=" * 70)


def main_interactive_mode():
    """
    Chế độ Interactive: Cho phép vẽ mask và xóa nhiều lần
    (Giống như code ban đầu của bạn)
    """
    image_path = "inputs/test_image2.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy file '{image_path}'")
        sys.exit(1)
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🎨 INTERACTIVE MODE - Vẽ và Xóa tự do")
    print("=" * 70)
    
    # Khởi tạo app
    app = IntelligentScissorsApp(image_path)
    inpainting_strategy = TraditionalInpainting(method='ns', radius=3)
    
    print("\n📋 HƯỚNG DẪN:")
    print("  🖱️  Chuột Trái  : Thêm điểm neo")
    print("  🖱️  Chuột Phải  : Kết thúc vòng vẽ")
    print("  ⌨️  ENTER       : Kết thúc vòng vẽ")
    print("  ⌨️  BACKSPACE   : Undo")
    print("  ⌨️  X           : XÓA vùng đã chọn (Inpainting)")
    print("  ⌨️  S           : Lưu Mask")
    print("  ⌨️  I           : Lưu ảnh hiện tại")
    print("  ⌨️  ESC         : Thoát")
    print("=" * 70 + "\n")
    
    app.update_display()
    
    while True:
        key = cv2.waitKey(20) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 13:  # ENTER
            if app.is_started:
                app.finish_drawing()
        elif key == 8:  # BACKSPACE
            app.undo_last_step()
        elif key == ord('x') or key == ord('X'):  # Xóa
            if cv2.countNonZero(app.global_mask) == 0:
                print("⚠️  Chưa có vùng nào được chọn!")
                continue
            
            print("🔄 Đang Inpainting...")
            app.img = inpainting_strategy.process(app.img, app.global_mask)
            app.global_mask[:] = 0
            app.tool.applyImage(app.img)
            print("✅ Đã xóa!")
            app.update_display()
        elif key == ord('s') or key == ord('S'):  # Lưu mask
            if cv2.countNonZero(app.global_mask) > 0:
                cv2.imwrite(os.path.join(output_dir, "mask.png"), app.global_mask)
                print("💾 Đã lưu mask!")
        elif key == ord('i') or key == ord('I'):  # Lưu ảnh
            cv2.imwrite(os.path.join(output_dir, "current_image.png"), app.img)
            print("💾 Đã lưu ảnh!")
    
    # Lưu ảnh cuối
    cv2.imwrite(os.path.join(output_dir, "final_result.png"), app.img)
    cv2.destroyAllWindows()
    print("✨ Hoàn thành!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Scissors + Inpainting")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pipeline', 'interactive'],
        default='pipeline',
        help='Chế độ chạy: pipeline (1 lần) hoặc interactive (nhiều lần)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'pipeline':
            # Chế độ Pipeline: Segmentation -> Inpainting -> Done
            main()
        else:
            # Chế độ Interactive: Vẽ và xóa tự do
            main_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã dừng bởi người dùng (Ctrl+C)")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)