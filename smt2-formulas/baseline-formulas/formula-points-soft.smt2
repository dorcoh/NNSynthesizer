; ---- baseline formula for soft points property (generated by hand)
(declare-fun weight_1_1_1 () Real)
(declare-fun X1 () Bool)
(declare-fun X2 () Bool)
(declare-fun X3 () Bool)
(declare-fun X4 () Bool)
(assert 
(and 
(forall ((input_1 Real) (input_2 Real))
  (let ((a!1 (ite (>= (+ (* input_1 weight_1_1_1)
                         (* input_2 (/ 6328408122062683.0 10000000000000000.0))
                         (/ 4678647518157959.0 10000000000000000.0))
                      0.0)
                  (+ (* input_1 weight_1_1_1)
                     (* input_2 (/ 6328408122062683.0 10000000000000000.0))
                     (/ 4678647518157959.0 10000000000000000.0))
                  0.0))
        (a!2 (ite (>= (+ (* input_1 (/ 1810682713985443.0 2500000000000000.0))
                         (* input_2
                            (- (/ 2702817916870117.0 5000000000000000.0)))
                         (/ 3324844539165497.0 10000000000000000.0))
                      0.0)
                  (+ (* input_1 (/ 1810682713985443.0 2500000000000000.0))
                     (* input_2 (- (/ 2702817916870117.0 5000000000000000.0)))
                     (/ 3324844539165497.0 10000000000000000.0))
                  0.0))
        (a!3 (ite (>= (+ (* input_1
                            (- (/ 756735622882843.0 2000000000000000.0)))
                         (* input_2 (/ 6610729098320007.0 10000000000000000.0))
                         (/ 895128607749939.0 1000000000000000.0))
                      0.0)
                  (+ (* input_1 (- (/ 756735622882843.0 2000000000000000.0)))
                     (* input_2 (/ 6610729098320007.0 10000000000000000.0))
                     (/ 895128607749939.0 1000000000000000.0))
                  0.0))
        (a!4 (ite (>= (+ (* input_1
                            (- (/ 3566634953022003.0 5000000000000000.0)))
                         (* input_2 (- (/ 66858921200037.0 78125000000000.0)))
                         (/ 49231022596359253.0 100000000000000000.0))
                      0.0)
                  (+ (* input_1 (- (/ 3566634953022003.0 5000000000000000.0)))
                     (* input_2 (- (/ 66858921200037.0 78125000000000.0)))
                     (/ 49231022596359253.0 100000000000000000.0))
                  0.0)))
  (let ((a!5 (+ (* a!1 (/ 6938807964324951.0 10000000000000000.0))
                (* a!2 (- (/ 1154119849205017.0 2000000000000000.0)))
                (* a!3 (- (/ 482571542263031.0 2000000000000000.0)))
                (* a!4 (/ 2846791036427021.0 25000000000000000.0))
                (/ 15994594991207123.0 50000000000000000.0)))
        (a!6 (+ (* a!1 (- (/ 7026025652885437.0 20000000000000000.0)))
                (* a!2 (/ 873803049325943.0 2500000000000000.0))
                (* a!3 (/ 1523522138595581.0 2000000000000000.0))
                (* a!4 (- (/ 1562498789280653.0 3125000000000000.0)))
                (/ 3052498959004879.0 12500000000000000.0))))
  	
    (=> (and (>= input_1 1.0)
                  (<= input_1 19.0)
                  (>= input_2 1.0)
                  (<= input_2 19.0))
             (and (> a!5 a!6)))
		 ))) ; end of forall
	
; ---- now we define a specific point. Note that it still uses weight_1_1_1:

((_ at-least 2)

    (let ((in_1 (+ (/ 5.0 2.0))) (in_2 (- (/ 5.0 2.0))))
        (let
            ((a!1_1 (ite (>= (+ (* in_1 weight_1_1_1)
                             (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                             (/ 4678647518157959.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 weight_1_1_1)
                         (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                         (/ 4678647518157959.0 10000000000000000.0))
                      0.0))
            (a!2_1 (ite (>= (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                             (* in_2
                                (- (/ 2702817916870117.0 5000000000000000.0)))
                             (/ 3324844539165497.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                         (* in_2 (- (/ 2702817916870117.0 5000000000000000.0)))
                         (/ 3324844539165497.0 10000000000000000.0))
                      0.0))
            (a!3_1 (ite (>= (+ (* in_1
                                (- (/ 756735622882843.0 2000000000000000.0)))
                             (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                             (/ 895128607749939.0 1000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 756735622882843.0 2000000000000000.0)))
                         (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                         (/ 895128607749939.0 1000000000000000.0))
                      0.0))
            (a!4_1 (ite (>= (+ (* in_1
                                (- (/ 3566634953022003.0 5000000000000000.0)))
                             (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                             (/ 49231022596359253.0 100000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 3566634953022003.0 5000000000000000.0)))
                         (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                         (/ 49231022596359253.0 100000000000000000.0))
                      0.0)))
      (let ((a!5_1 (+ (* a!1_1 (/ 6938807964324951.0 10000000000000000.0))
                    (* a!2_1 (- (/ 1154119849205017.0 2000000000000000.0)))
                    (* a!3_1 (- (/ 482571542263031.0 2000000000000000.0)))
                    (* a!4_1 (/ 2846791036427021.0 25000000000000000.0))
                    (/ 15994594991207123.0 50000000000000000.0)))
            (a!6_1 (+ (* a!1_1 (- (/ 7026025652885437.0 20000000000000000.0)))
                    (* a!2_1 (/ 873803049325943.0 2500000000000000.0))
                    (* a!3_1 (/ 1523522138595581.0 2000000000000000.0))
                    (* a!4_1 (- (/ 1562498789280653.0 3125000000000000.0)))
                    (/ 3052498959004879.0 12500000000000000.0))))
            (> a!6_1 a!5_1) ; becomes sat if we swap to <
        )
    ))

    (let ((in_1 (+ (/ 15 2.0))) (in_2 (- (/ 5.0 2.0))))
        (let
            ((a!1_1 (ite (>= (+ (* in_1 weight_1_1_1)
                             (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                             (/ 4678647518157959.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 weight_1_1_1)
                         (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                         (/ 4678647518157959.0 10000000000000000.0))
                      0.0))
            (a!2_1 (ite (>= (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                             (* in_2
                                (- (/ 2702817916870117.0 5000000000000000.0)))
                             (/ 3324844539165497.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                         (* in_2 (- (/ 2702817916870117.0 5000000000000000.0)))
                         (/ 3324844539165497.0 10000000000000000.0))
                      0.0))
            (a!3_1 (ite (>= (+ (* in_1
                                (- (/ 756735622882843.0 2000000000000000.0)))
                             (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                             (/ 895128607749939.0 1000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 756735622882843.0 2000000000000000.0)))
                         (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                         (/ 895128607749939.0 1000000000000000.0))
                      0.0))
            (a!4_1 (ite (>= (+ (* in_1
                                (- (/ 3566634953022003.0 5000000000000000.0)))
                             (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                             (/ 49231022596359253.0 100000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 3566634953022003.0 5000000000000000.0)))
                         (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                         (/ 49231022596359253.0 100000000000000000.0))
                      0.0)))
      (let ((a!5_1 (+ (* a!1_1 (/ 6938807964324951.0 10000000000000000.0))
                    (* a!2_1 (- (/ 1154119849205017.0 2000000000000000.0)))
                    (* a!3_1 (- (/ 482571542263031.0 2000000000000000.0)))
                    (* a!4_1 (/ 2846791036427021.0 25000000000000000.0))
                    (/ 15994594991207123.0 50000000000000000.0)))
            (a!6_1 (+ (* a!1_1 (- (/ 7026025652885437.0 20000000000000000.0)))
                    (* a!2_1 (/ 873803049325943.0 2500000000000000.0))
                    (* a!3_1 (/ 1523522138595581.0 2000000000000000.0))
                    (* a!4_1 (- (/ 1562498789280653.0 3125000000000000.0)))
                    (/ 3052498959004879.0 12500000000000000.0))))
            (> a!6_1 a!5_1) ; becomes sat if we swap to <
        )
    ))

    (let ((in_1 (+ (/ 25.0 2.0))) (in_2 (- (/ 5.0 2.0))))
        (let
            ((a!1_1 (ite (>= (+ (* in_1 weight_1_1_1)
                             (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                             (/ 4678647518157959.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 weight_1_1_1)
                         (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                         (/ 4678647518157959.0 10000000000000000.0))
                      0.0))
            (a!2_1 (ite (>= (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                             (* in_2
                                (- (/ 2702817916870117.0 5000000000000000.0)))
                             (/ 3324844539165497.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                         (* in_2 (- (/ 2702817916870117.0 5000000000000000.0)))
                         (/ 3324844539165497.0 10000000000000000.0))
                      0.0))
            (a!3_1 (ite (>= (+ (* in_1
                                (- (/ 756735622882843.0 2000000000000000.0)))
                             (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                             (/ 895128607749939.0 1000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 756735622882843.0 2000000000000000.0)))
                         (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                         (/ 895128607749939.0 1000000000000000.0))
                      0.0))
            (a!4_1 (ite (>= (+ (* in_1
                                (- (/ 3566634953022003.0 5000000000000000.0)))
                             (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                             (/ 49231022596359253.0 100000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 3566634953022003.0 5000000000000000.0)))
                         (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                         (/ 49231022596359253.0 100000000000000000.0))
                      0.0)))
      (let ((a!5_1 (+ (* a!1_1 (/ 6938807964324951.0 10000000000000000.0))
                    (* a!2_1 (- (/ 1154119849205017.0 2000000000000000.0)))
                    (* a!3_1 (- (/ 482571542263031.0 2000000000000000.0)))
                    (* a!4_1 (/ 2846791036427021.0 25000000000000000.0))
                    (/ 15994594991207123.0 50000000000000000.0)))
            (a!6_1 (+ (* a!1_1 (- (/ 7026025652885437.0 20000000000000000.0)))
                    (* a!2_1 (/ 873803049325943.0 2500000000000000.0))
                    (* a!3_1 (/ 1523522138595581.0 2000000000000000.0))
                    (* a!4_1 (- (/ 1562498789280653.0 3125000000000000.0)))
                    (/ 3052498959004879.0 12500000000000000.0))))
            (> a!6_1 a!5_1) ; becomes sat if we swap to <
        )
    ))

    (let ((in_1 (+ (/ 35.0 2.0))) (in_2 (- (/ 5.0 2.0))))
        (let
            ((a!1_1 (ite (>= (+ (* in_1 weight_1_1_1)
                             (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                             (/ 4678647518157959.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 weight_1_1_1)
                         (* in_2 (/ 6328408122062683.0 10000000000000000.0))
                         (/ 4678647518157959.0 10000000000000000.0))
                      0.0))
            (a!2_1 (ite (>= (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                             (* in_2
                                (- (/ 2702817916870117.0 5000000000000000.0)))
                             (/ 3324844539165497.0 10000000000000000.0))
                          0.0)
                      (+ (* in_1 (/ 1810682713985443.0 2500000000000000.0))
                         (* in_2 (- (/ 2702817916870117.0 5000000000000000.0)))
                         (/ 3324844539165497.0 10000000000000000.0))
                      0.0))
            (a!3_1 (ite (>= (+ (* in_1
                                (- (/ 756735622882843.0 2000000000000000.0)))
                             (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                             (/ 895128607749939.0 1000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 756735622882843.0 2000000000000000.0)))
                         (* in_2 (/ 6610729098320007.0 10000000000000000.0))
                         (/ 895128607749939.0 1000000000000000.0))
                      0.0))
            (a!4_1 (ite (>= (+ (* in_1
                                (- (/ 3566634953022003.0 5000000000000000.0)))
                             (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                             (/ 49231022596359253.0 100000000000000000.0))
                          0.0)
                      (+ (* in_1 (- (/ 3566634953022003.0 5000000000000000.0)))
                         (* in_2 (- (/ 66858921200037.0 78125000000000.0)))
                         (/ 49231022596359253.0 100000000000000000.0))
                      0.0)))
      (let ((a!5_1 (+ (* a!1_1 (/ 6938807964324951.0 10000000000000000.0))
                    (* a!2_1 (- (/ 1154119849205017.0 2000000000000000.0)))
                    (* a!3_1 (- (/ 482571542263031.0 2000000000000000.0)))
                    (* a!4_1 (/ 2846791036427021.0 25000000000000000.0))
                    (/ 15994594991207123.0 50000000000000000.0)))
            (a!6_1 (+ (* a!1_1 (- (/ 7026025652885437.0 20000000000000000.0)))
                    (* a!2_1 (/ 873803049325943.0 2500000000000000.0))
                    (* a!3_1 (/ 1523522138595581.0 2000000000000000.0))
                    (* a!4_1 (- (/ 1562498789280653.0 3125000000000000.0)))
                    (/ 3052498959004879.0 12500000000000000.0))))
            (> a!6_1 a!5_1) ; becomes sat if we swap to <
        )
    ))
)
))
(check-sat)
(get-model)
