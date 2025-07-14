import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

# Initialiser MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class FaceSwapper:
    def __init__(self, reference_image_path):
        self.reference_image = None
        self.reference_landmarks = None
        self.reference_points = None
        self.triangulation = None
        self.adapted_reference = None  # Version adaptée de l'image de référence
        self.target_brightness = None  # Luminosité cible à adapter
        
        # Charger l'image de référence
        self.load_reference_face(reference_image_path)
        
        # Initialiser Face Mesh pour la vidéo
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_and_crop_face(self, image):
        """Détecte le visage et découpe l'image autour"""
        # Utiliser MediaPipe pour détecter le visage
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3  # Plus permissif pour la détection initiale
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w = image.shape[:2]
                
                # Calculer la boîte englobante du visage
                x_coords = [int(landmark.x * w) for landmark in landmarks.landmark]
                y_coords = [int(landmark.y * h) for landmark in landmarks.landmark]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Ajouter une marge autour du visage (20% de chaque côté)
                margin_x = int((x_max - x_min) * 0.2)
                margin_y = int((y_max - y_min) * 0.2)
                
                # Calculer les coordonnées finales avec marges
                crop_x_min = max(0, x_min - margin_x)
                crop_y_min = max(0, y_min - margin_y)
                crop_x_max = min(w, x_max + margin_x)
                crop_y_max = min(h, y_max + margin_y)
                
                # Découper l'image
                face_crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                
                print(f"Visage détecté: boîte ({crop_x_min}, {crop_y_min}) -> ({crop_x_max}, {crop_y_max})")
                
                return face_crop
            
            return None
        
    def load_reference_face(self, image_path):
        """Charge l'image de référence et extrait ses landmarks"""
        print(f"Chargement de l'image de référence: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Erreur: Impossible de charger {image_path}")
            return False
        
        # Détecter le visage pour découper la zone d'intérêt

        face_region = self.detect_and_crop_face(image)
        
        if face_region is None:
            print("❌ Aucun visage détecté pour le découpage, utilisation de l'image complète")
            face_region = image
        else:
            print(f"✅ Visage détecté et découpé: {face_region.shape[1]}x{face_region.shape[0]}")
        
        # Redimensionner si la région du visage est encore trop grande
        H, W = face_region.shape[:2]
        max_size = 100  # Taille maximale pour les performances
        
        if H > max_size or W > max_size:
            if W > H:
                new_w = max_size
                new_h = int(H * max_size / W)
            else:
                new_h = max_size
                new_w = int(W * max_size / H)
            
            face_region = cv2.resize(face_region, (new_w, new_h))
            print(f"Région du visage redimensionnée: {new_w}x{new_h}")
    
        self.reference_image = face_region
        h, w = face_region.shape[:2]
        image_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Extraire les landmarks de l'image de référence
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                self.reference_landmarks = results.multi_face_landmarks[0]
                
                # Convertir les landmarks en points 2D
                self.reference_points = []
                for landmark in self.reference_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    self.reference_points.append([x, y])
                
                self.reference_points = np.array(self.reference_points, dtype=np.float32)
                
                # Créer la triangulation
                self.create_triangulation()
                
                print(f"✅ Landmarks extraits: {len(self.reference_landmarks.landmark)} points")
                return True
            else:
                print("❌ Aucun visage détecté dans l'image de référence")
                return False
    
    def create_triangulation(self):
        """Crée la triangulation de Delaunay pour les points du visage uniquement
        La triangulaiton de Delaunay est une methode pour diviser un ensemble de points en triangles de manière optimale."""
        if self.reference_points is None:
            return
        
        # Utiliser seulement les points du visage, sans les coins de l'image
        try:
            tri = Delaunay(self.reference_points)
            self.triangulation = tri.simplices
            print(f"✅ Triangulation créée: {len(self.triangulation)} triangles pour le visage")
        except Exception as e:
            print(f"Erreur lors de la triangulation: {e}")
            self.triangulation = None
    
    def get_triangle_transform(self, src_triangle, dst_triangle):
        """Calcule la transformation affine entre deux triangles"""
        # Convertir en format requis pour getAffineTransform
        src = np.float32(src_triangle)
        dst = np.float32(dst_triangle)
        
        # Calculer la transformation affine
        transform_matrix = cv2.getAffineTransform(src, dst)
        return transform_matrix
    
    def warp_triangle(self, src_img, dst_img, src_triangle, dst_triangle):
        """Applique la transformation à un triangle"""
        # Obtenir les boîtes englobantes
        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dst_rect = cv2.boundingRect(np.float32([dst_triangle]))
        
        # Vérifier que les rectangles sont valides
        if (src_rect[2] <= 0 or src_rect[3] <= 0 or 
            dst_rect[2] <= 0 or dst_rect[3] <= 0):
            return
        
        # Vérifier les limites de l'image
        src_h, src_w = src_img.shape[:2]
        dst_h, dst_w = dst_img.shape[:2]
        
        if (src_rect[0] >= src_w or src_rect[1] >= src_h or
            dst_rect[0] >= dst_w or dst_rect[1] >= dst_h):
            return
        
        # Ajuster les rectangles pour rester dans les limites
        src_rect = (max(0, src_rect[0]), max(0, src_rect[1]),
                   min(src_rect[2], src_w - max(0, src_rect[0])),
                   min(src_rect[3], src_h - max(0, src_rect[1])))
        
        dst_rect = (max(0, dst_rect[0]), max(0, dst_rect[1]),
                   min(dst_rect[2], dst_w - max(0, dst_rect[0])),
                   min(dst_rect[3], dst_h - max(0, dst_rect[1])))
        
        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            return
        
        # Décaler les points du triangle par rapport aux rectangles
        src_tri_rect = []
        dst_tri_rect = []
        
        for i in range(3):
            src_tri_rect.append(((src_triangle[i][0] - src_rect[0]), 
                               (src_triangle[i][1] - src_rect[1])))
            dst_tri_rect.append(((dst_triangle[i][0] - dst_rect[0]), 
                               (dst_triangle[i][1] - dst_rect[1])))
        
        # Extraire la région source
        try:
            src_crop = src_img[src_rect[1]:src_rect[1] + src_rect[3], 
                              src_rect[0]:src_rect[0] + src_rect[2]]
            
            if src_crop.size == 0 or src_crop.shape[0] == 0 or src_crop.shape[1] == 0:
                return
            
            # Calculer la transformation affine
            transform = cv2.getAffineTransform(np.float32(src_tri_rect), 
                                             np.float32(dst_tri_rect))
            
            # Appliquer la transformation
            dst_crop = cv2.warpAffine(src_crop, transform, (dst_rect[2], dst_rect[3]))
            
            if dst_crop.size == 0:
                return
            
            # Créer un masque pour le triangle destination
            mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(dst_tri_rect), (1.0, 1.0, 1.0))
            
            # Appliquer le masque
            dst_crop = dst_crop * mask
            
            # Région de destination dans l'image finale
            dst_region = dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], 
                               dst_rect[0]:dst_rect[0] + dst_rect[2]]
            
            # Vérifier que les dimensions correspondent
            if (dst_region.shape[:2] == dst_crop.shape[:2] and 
                dst_region.shape[:2] == mask.shape[:2]):
                # Mélanger avec l'image existante
                dst_region[:] = dst_region * (1.0 - mask) + dst_crop
                
        except Exception as e:
            # Ignorer les erreurs de transformation et continuer
            pass
    
    def swap_faces(self, target_image, target_landmarks, enable_color_adaptation=True):
        """Effectue le swap de visage"""
        if (self.reference_image is None or 
            self.reference_points is None or 
            self.triangulation is None):
            return target_image
        
        # Utiliser la référence adaptée si disponible, sinon l'originale
        working_reference = self.adapted_reference if self.adapted_reference is not None else self.reference_image
        
        h, w = target_image.shape[:2]
        result = target_image.copy().astype(np.float32)
        
        # Convertir les landmarks cibles en points 2D
        target_points = []
        for landmark in target_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            target_points.append([x, y])
        
        target_points = np.array(target_points, dtype=np.float32)
        
        # Créer un masque pour délimiter la zone du visage
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Définir les points du contour du visage (utilise les landmarks du contour)
        face_outline_indices = [
            # Contour du visage
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        if len(target_points) >= max(face_outline_indices):
            face_contour = target_points[face_outline_indices].astype(np.int32)
            cv2.fillPoly(face_mask, [face_contour], 255)
            
            
            inner_mouth_indices = [
                # bouche intérieure
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
                310, 311, 312, 13, 82, 81, 80, 78
            ]
            
            if len(target_points) >= max(inner_mouth_indices):
                inner_mouth_contour = target_points[inner_mouth_indices].astype(np.int32)
                cv2.fillPoly(face_mask, [inner_mouth_contour], 0)  # Mettre à 0 pour exclure
            
            # Agrandir légèrement la zone intérieure de la bouche exclue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            
            # Créer un masque séparé pour l'intérieur de la bouche
            inner_mouth_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Recréer la zone intérieure de la bouche dans le masque séparé
            if len(target_points) >= max(inner_mouth_indices):
                inner_mouth_contour = target_points[inner_mouth_indices].astype(np.int32)
                cv2.fillPoly(inner_mouth_mask, [inner_mouth_contour], 255)
            
            # Dilater légèrement la zone à exclure
            inner_mouth_mask = cv2.dilate(inner_mouth_mask, kernel, iterations=1)
            
            # Soustraire la zone dilatée du masque principal
            face_mask = cv2.subtract(face_mask, inner_mouth_mask)
            
            # Appliquer un flou au masque pour une transition plus douce
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
            face_mask = face_mask.astype(np.float32) / 255.0
        else:
            # Si on n'a pas assez de landmarks, créer un masque elliptique
            center_x = int(np.mean(target_points[:, 0]))
            center_y = int(np.mean(target_points[:, 1]))
            
            # Calculer les dimensions approximatives du visage
            face_width = int(np.max(target_points[:, 0]) - np.min(target_points[:, 0]))
            face_height = int(np.max(target_points[:, 1]) - np.min(target_points[:, 1]))
            
            cv2.ellipse(face_mask, (center_x, center_y), 
                       (face_width//2, face_height//2), 0, 0, 360, 255, -1)
            
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
            face_mask = face_mask.astype(np.float32) / 255.0
        
        # Utiliser les triangles MediaPipe prédéfinis
        ref_points_face = self.reference_points
        target_points_face = target_points
        
        # Créer une triangulation spécifique pour les points du visage
        try:
            tri = Delaunay(ref_points_face)
            triangulation_face = tri.simplices
        except:
            return target_image
        
        
        # Appliquer la transformation pour chaque triangle du visage uniquement
        for triangle in triangulation_face:
            if all(idx < len(target_points_face) and idx < len(ref_points_face) for idx in triangle):
                src_triangle = ref_points_face[triangle]
                dst_triangle = target_points_face[triangle]
                
                # Utiliser la référence adaptée au lieu de l'originale
                self.warp_triangle(working_reference.astype(np.float32), 
                                 result, src_triangle, dst_triangle)
        
        # SUPPRESSION: Plus besoin d'adaptation en temps réel
        # L'adaptation a déjà été faite dans working_reference
        
        # Mélanger le résultat avec l'image originale en utilisant le masque
        result_uint8 = result.astype(np.uint8)
        target_uint8 = target_image.astype(np.uint8)
        
        # Appliquer le masque pour ne garder que la zone du visage
        for c in range(3):  # Pour chaque canal de couleur
            result_uint8[:, :, c] = (face_mask * result_uint8[:, :, c] + 
                                   (1 - face_mask) * target_uint8[:, :, c])
        
        return result_uint8
    
    def process_frame(self, frame, frame_count=0):
        """Traite une frame de la vidéo"""
        if self.reference_image is None:
            return frame
        
        # Mettre à jour la luminosité cible toutes les secondes (30 FPS = 30 frames/seconde)
        if frame_count % 3 == 0:
            self.update_target_brightness(frame)
        
        # Convertir en RGB pour MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détecter les landmarks du visage
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Effectuer le swap de visage (plus besoin d'adaptation en temps réel)
                swapped_frame = self.swap_faces(frame, face_landmarks, enable_color_adaptation=False)
                return swapped_frame
        
        return frame

    
    def analyze_reference_brightness(self):
        """Analyse la luminosité de l'image de référence une seule fois"""
        if self.reference_image is None:
            return None
        
        try:
            # Convertir en LAB pour analyse de luminosité
            ref_lab = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2LAB)
            
            # Analyser seulement la luminosité moyenne
            ref_brightness = np.mean(ref_lab[:, :, 0])
            ref_a_mean = np.mean(ref_lab[:, :, 1])
            ref_b_mean = np.mean(ref_lab[:, :, 2])
            
            print(f"🔍 Luminosité référence analysée: L={ref_brightness:.1f}, A={ref_a_mean:.1f}, B={ref_b_mean:.1f}")
            
            return {
                'brightness': ref_brightness,
                'a_mean': ref_a_mean,
                'b_mean': ref_b_mean
            }
        except Exception as e:
            print(f"Erreur analyse référence: {e}")
            return None

    def create_adapted_reference(self, target_brightness_stats):
        """Crée une version adaptée de l'image de référence basée sur la luminosité cible"""
        if self.reference_image is None or target_brightness_stats is None:
            self.adapted_reference = self.reference_image
            return
        
        try:
            ref_lab = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Calculer les facteurs d'adaptation
            ref_stats = self.analyze_reference_brightness()
            if ref_stats is None:
                self.adapted_reference = self.reference_image
                return
            
            # Facteurs d'ajustement
            l_factor = target_brightness_stats['brightness'] / ref_stats['brightness'] if ref_stats['brightness'] > 0 else 1.0
            a_shift = target_brightness_stats['a_mean'] - ref_stats['a_mean']
            b_shift = target_brightness_stats['b_mean'] - ref_stats['b_mean']
            
            # Limiter les ajustements
            l_factor = np.clip(l_factor, 0.7, 1.4)  # Plus conservateur
            a_shift = np.clip(a_shift, -15, 15)
            b_shift = np.clip(b_shift, -15, 15)
            
            # Appliquer l'adaptation
            adapted_lab = ref_lab.copy()
            adapted_lab[:, :, 0] = np.clip(adapted_lab[:, :, 0] * l_factor, 0, 255)
            adapted_lab[:, :, 1] = np.clip(adapted_lab[:, :, 1] + a_shift, 0, 255)
            adapted_lab[:, :, 2] = np.clip(adapted_lab[:, :, 2] + b_shift, 0, 255)
            
            # Reconvertir en BGR
            self.adapted_reference = cv2.cvtColor(adapted_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
            print(f"✨ Référence adaptée créée: facteur={l_factor:.2f}, décalage=({a_shift:.1f}, {b_shift:.1f})")
            
        except Exception as e:
            print(f"Erreur création référence adaptée: {e}")
            self.adapted_reference = self.reference_image

    def update_target_brightness(self, frame):
        """Met à jour la luminosité cible basée sur la frame actuelle (appelé rarement)"""
        try:
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Analyser seulement une zone centrale de la frame
            frame_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped=self.detect_and_crop_face(frame_temp)
            frame_lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
            h, w = frame_lab.shape[:2]
            center_region = frame_lab[h//4:3*h//4, w//4:3*w//4]
            
            target_stats = {
                'brightness': np.mean(center_region[:, :, 0]),
                'a_mean': np.mean(center_region[:, :, 1]),
                'b_mean': np.mean(center_region[:, :, 2])
            }
            
            # Créer la référence adaptée seulement si elle n'existe pas ou si le changement est significatif
            if (self.target_brightness is None or 
                abs(self.target_brightness['brightness'] - target_stats['brightness']) > 20):
                
                self.target_brightness = target_stats
                self.create_adapted_reference(target_stats)
                print(f"🎯 Luminosité cible mise à jour: {target_stats['brightness']:.1f}")
                
        except Exception as e:
            print(f"Erreur mise à jour luminosité: {e}")

def main():
    # Initialiser le face swapper
    face_swapper = FaceSwapper("image/head.png")
    
    if face_swapper.reference_image is None:
        print("Erreur: Impossible de charger l'image de référence")
        return
    
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam")
        return
    
    print("Face Swapper initialisé! Appuyez sur 'q' pour quitter, 's' pour sauvegarder une frame")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Traiter la frame avec le compteur
            result_frame = face_swapper.process_frame(frame, frame_count)
            frame_count += 1
            
            # Afficher le résultat
            cv2.imshow('Face Swap ', cv2.flip(result_frame, 1))
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Sauvegarder la frame actuelle
                filename = f"face_swap_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Frame sauvegardée: {filename}")
            
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Nettoyage terminé")

if __name__ == "__main__":
    main()
