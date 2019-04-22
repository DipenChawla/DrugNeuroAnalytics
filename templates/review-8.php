<?php
include('./Classes/DB.php');
$drug_name = 'Valporic Acid';
$reviews = DB::query('SELECT * FROM Reviews WHERE Drug=:drug', array(':drug'=>$drug_name));
// var_dump($reviews);

foreach ($reviews as $review ) {

  // code...

  echo $review['Review'] . '<br>';
}

 ?>
