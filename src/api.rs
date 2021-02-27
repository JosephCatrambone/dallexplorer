use tide::prelude::*;

use crate::{State, image_to_tensor, LATENT_SIZE};

#[derive(Debug, Deserialize)]
pub struct ImageCrossRequest {
	latent_a: Vec<u8>,
	latent_b: Vec<u8>,
	num_children: u8
}

//async fn generate_image(mut req: tide::Request<()>) -> tide::Result {
pub async fn encode_image(mut req: tide::Request<State>) -> tide::Result {
	//let ImageEncodeRequest { image } = req.body_json().await?;
	let image_file = req.body_bytes().await?;

	// Load and error check.
	let img = image::load_from_memory(&image_file);
	if img.is_err() {
		return tide::Result::Err(img.unwrap_err().into());
	}
	let img = img.unwrap();

	let state = req.state();
	let repr = state.image_to_vec(&img);

	let response = format!("{:?}", &repr);
	dbg!("{:?}", &response);

	Ok(response.into())
}
